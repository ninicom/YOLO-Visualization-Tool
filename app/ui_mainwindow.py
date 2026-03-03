import sys
import os
import shutil
import cv2
import torch
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QProgressBar, QFileDialog,
    QGridLayout, QScrollArea, QFrame, QLineEdit, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QMessageBox, QTabWidget, QSlider,
    QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QIntValidator
from ultralytics import YOLO
from workers import GradCamWorker, VideoWorker # Import workers

class MainWindow(QMainWindow):
    
    # Định nghĩa các trạng thái
    STATE_STARTUP = 0
    STATE_IDLE = 1
    STATE_IMAGE_LOADED = 2
    STATE_VIDEO_PAUSED = 3
    STATE_VIDEO_RUNNING = 4
    STATE_GRADCAM_RUNNING = 5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grad-CAM Visualizer cho YOLO")
        self.setGeometry(100, 100, 1400, 900)
        
        # Biến lưu trữ
        self.model = None
        self.current_model_path = None
        self.class_names = {}
        self.image_files_list = []
        self.current_image_index = -1
        self.current_grid_pixmap = None
        self.current_image_pixmap = None
        
        # Biến video
        self.video_worker = None
        self.current_video_path = None
        self.current_video_frame = None # Frame gốc (np.ndarray)
        self.video_capture = None
        self.video_total_frames = 0
        self.video_current_frame_idx = 0
        self.slider_is_pressed = False # Cờ cho thanh slider
        self.detect_timer = QTimer(self) # Timer cho việc detect (debounce)
        self.detect_timer.setSingleShot(True)
        self.detect_timer.timeout.connect(self.run_detection_on_current_frame)

        # Biến trạng thái
        self.current_state = self.STATE_STARTUP
        self.previous_state = self.STATE_STARTUP
        
        # Thư mục tạm
        self.temp_dir = "temp_gradcam_output"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.temp_frame_path = os.path.join(self.temp_dir, "current_video_frame.jpg")

        # --- Tạo giao diện ---
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        content_layout = QHBoxLayout()
        
        self.control_scroll_area = self.create_control_panel()
        content_layout.addWidget(self.control_scroll_area)
        
        display_widget = self.create_display_area()
        content_layout.addWidget(display_widget, 1)
        
        main_layout.addLayout(content_layout)
        self.setCentralWidget(main_widget)
        
        # --- Tải model mặc định & Cập nhật UI ---
        QTimer.singleShot(100, self.load_default_model)
        self.set_ui_for_state(self.STATE_STARTUP)

    # --- 1. HÀM TẠO GIAO DIỆN ---
    def create_control_panel(self):
        control_panel_widget = QWidget()
        control_layout = QVBoxLayout(control_panel_widget)
        
        # --- Nhóm 1: Tải Model & Lớp ---
        group1_box = QFrame()
        group1_box.setFrameShape(QFrame.Shape.StyledPanel)
        group1_layout = QVBoxLayout(group1_box)
        group1_layout.addWidget(QLabel("--- 1. Tải Model & Lớp ---"))
        
        self.btn_select_model = QPushButton("Chọn file Model (.pt) khác")
        self.btn_select_model.clicked.connect(self.on_select_model_clicked)
        group1_layout.addWidget(self.btn_select_model)
        
        group1_layout.addWidget(QLabel("Model đang tải:"))
        self.model_path_label = QLabel("Chưa chọn model.")
        self.model_path_label.setStyleSheet("font-style: italic; color: gray;")
        self.model_path_label.setWordWrap(True)
        group1_layout.addWidget(self.model_path_label)
        
        group1_layout.addWidget(QLabel("Chọn Lớp để chạy Grad-CAM:"))
        self.combo_classes = QComboBox()
        self.combo_classes.addItem("Tải model để xem lớp", -1)
        self.combo_classes.addItem("Tự động phát hiện", -2)
        group1_layout.addWidget(self.combo_classes)
        control_layout.addWidget(group1_box)

        # --- Nhóm 2: Cấu hình Grad-CAM & Detect ---
        self.params_panel = QFrame() # Lưu con trỏ
        self.params_panel.setFrameShape(QFrame.Shape.StyledPanel)
        group2_layout = QVBoxLayout(self.params_panel)
        group2_layout.addWidget(QLabel("--- 2. Cấu hình Grad-CAM & Detect ---"))

        group2_layout.addWidget(QLabel("Phương thức (Method):"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(['GradCAM', 'GradCAMPlusPlus', 'XGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'RandomCAM', 'EigenGradCAM', 'KPCA_CAM', 'AblationCAM'])
        group2_layout.addWidget(self.combo_method)
        
        group2_layout.addWidget(QLabel("Tác vụ (Task):"))
        self.combo_task = QComboBox()
        self.combo_task.addItems(['detect', 'segment', 'pose', 'obb', 'classify'])
        self.combo_task.currentTextChanged.connect(self.update_ui_for_task) # Kết nối signal
        group2_layout.addWidget(self.combo_task)

        group2_layout.addWidget(QLabel("Loại Backward (Backward Type):"))
        self.combo_backward_type = QComboBox()
        self.combo_backward_type.addItems(['all', 'class', 'box', 'segment', 'keypoint', 'angle'])
        group2_layout.addWidget(self.combo_backward_type)
        
        group2_layout.addWidget(QLabel("Thiết bị (Device):"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(['cpu'])
        if torch.cuda.is_available():
            self.combo_device.addItems(['cuda'])
            self.combo_device.setCurrentText('cuda')
        else:
            self.combo_device.setEnabled(False)
        group2_layout.addWidget(self.combo_device)

        group2_layout.addWidget(QLabel("Layers (ví dụ: 16, 19, 22):"))
        self.edit_layers = QLineEdit("16, 19, 22")
        group2_layout.addWidget(self.edit_layers)
        
        group2_layout.addWidget(QLabel("Ngưỡng Conf (Detect Video & CAM):"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.4)
        group2_layout.addWidget(self.spin_conf)

        group2_layout.addWidget(QLabel("Tỷ lệ Ratio (cho target CAM):"))
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.01, 1.0); self.spin_ratio.setSingleStep(0.01); self.spin_ratio.setValue(0.02)
        group2_layout.addWidget(self.spin_ratio)

        group2_layout.addWidget(QLabel("Kích thước ảnh (Img Size - Detect Video & CAM):"))
        self.spin_img_size = QSpinBox()
        self.spin_img_size.setRange(64, 2048); self.spin_img_size.setSingleStep(32); self.spin_img_size.setValue(640)
        group2_layout.addWidget(self.spin_img_size)

        self.check_renormalize = QCheckBox("Chuẩn hóa trong Box (Renormalize CAM)")
        group2_layout.addWidget(self.check_renormalize)
        
        self.check_show_boxes = QCheckBox("Hiển thị Box (Detect Video & CAM)")
        self.check_show_boxes.setChecked(True)
        group2_layout.addWidget(self.check_show_boxes)
        control_layout.addWidget(self.params_panel)
        
        # --- Nhóm 3: Chạy và Lưu ---
        group3_box = QFrame()
        group3_box.setFrameShape(QFrame.Shape.StyledPanel)
        group3_layout = QVBoxLayout(group3_box)
        group3_layout.addWidget(QLabel("--- 3. Chạy và Lưu ---"))

        self.btn_reload_detect = QPushButton("🔄 Tải lại Detect (Ảnh/Frame)")
        self.btn_reload_detect.setToolTip("Chạy lại detection trên ảnh/frame hiện tại với cấu hình mới (ngưỡng, kích cỡ ảnh,...)")
        self.btn_reload_detect.clicked.connect(self.on_reload_detect_clicked)
        group3_layout.addWidget(self.btn_reload_detect)
        
        self.btn_run_gradcam = QPushButton("🔥 Chạy Grad-CAM (trên Ảnh/Frame)")
        self.btn_run_gradcam.clicked.connect(self.run_gradcam)
        group3_layout.addWidget(self.btn_run_gradcam)
        
        self.btn_save_grid = QPushButton("💾 Lưu Ảnh Grid")
        self.btn_save_grid.clicked.connect(self.save_grid)
        group3_layout.addWidget(self.btn_save_grid)
        
        group3_layout.addWidget(QLabel("Tiến độ:"))
        self.progress_bar = QProgressBar()
        group3_layout.addWidget(self.progress_bar)
        
        group3_layout.addWidget(QLabel("Trạng thái:"))
        self.log_label = QLabel("Khởi động...")
        self.log_label.setWordWrap(True)
        self.log_label.setMinimumHeight(60)
        self.log_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px;")
        group3_layout.addWidget(self.log_label)
        control_layout.addWidget(group3_box)

        control_layout.addStretch()
        
        control_scroll_area = QScrollArea()
        control_scroll_area.setWidgetResizable(True)
        control_scroll_area.setWidget(control_panel_widget)
        control_scroll_area.setFixedWidth(350)
        return control_scroll_area
    
    def create_display_area(self):
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        self.tabs = QTabWidget()
        self.tab_image = QWidget()
        self.tab_video = QWidget()
        self.tab_gradcam = QWidget()
        
        self.tabs.addTab(self.tab_image, "🖼️ Ảnh Tĩnh")
        self.tabs.addTab(self.tab_video, "🎬 Video")
        self.tabs.addTab(self.tab_gradcam, "🔥 Kết quả Grad-CAM")
        self.tabs.currentChanged.connect(self.on_tab_changed) # Kết nối sự kiện đổi tab
        
        display_layout.addWidget(self.tabs)

        # --- Tab 1: Ảnh Tĩnh ---
        tab_image_layout = QVBoxLayout(self.tab_image)
        image_toolbar = QHBoxLayout()
        self.btn_open_image = QPushButton("Chọn 1 Ảnh")
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_dir = QPushButton("Chọn Thư Mục Ảnh")
        self.btn_open_dir.clicked.connect(self.open_directory)
        self.btn_prev_img = QPushButton("<< Ảnh Trước")
        self.btn_prev_img.clicked.connect(self.prev_image)
        self.btn_next_img = QPushButton("Ảnh Tiếp >>")
        self.btn_next_img.clicked.connect(self.next_image)
        image_toolbar.addWidget(self.btn_open_image)
        image_toolbar.addWidget(self.btn_open_dir)
        image_toolbar.addStretch()
        image_toolbar.addWidget(self.btn_prev_img)
        image_toolbar.addWidget(self.btn_next_img)
        tab_image_layout.addLayout(image_toolbar)
        
        self.image_display_label = QLabel("Chọn ảnh để hiển thị")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setFrameShape(QFrame.Shape.Box)
        self.image_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) # Cho phép co giãn
        tab_image_layout.addWidget(self.image_display_label, 1)

        # --- Tab 2: Video ---
        tab_video_layout = QVBoxLayout(self.tab_video)
        video_toolbar = QHBoxLayout()
        self.btn_open_video = QPushButton("Mở Video")
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_play_pause = QPushButton("▶️ Play")
        self.btn_play_pause.clicked.connect(self.on_play_pause_clicked)
        self.btn_stop_video = QPushButton("⏹️ Stop")
        self.btn_stop_video.clicked.connect(self.stop_video_worker)
        video_toolbar.addWidget(self.btn_open_video)
        video_toolbar.addWidget(self.btn_play_pause)
        video_toolbar.addWidget(self.btn_stop_video)
        video_toolbar.addStretch()
        tab_video_layout.addLayout(video_toolbar)
        
        # Thanh trượt video
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.sliderPressed.connect(self.on_slider_pressed)
        self.video_slider.sliderReleased.connect(self.on_slider_released)
        # self.video_slider.valueChanged.connect(self.on_slider_moved) # Gây lag
        tab_video_layout.addWidget(self.video_slider)

        # Thanh điều khiển frame
        frame_toolbar = QHBoxLayout()
        self.btn_prev_frame = QPushButton("<<")
        self.btn_prev_frame.clicked.connect(self.on_prev_frame_clicked)
        self.btn_next_frame = QPushButton(">>")
        self.btn_next_frame.clicked.connect(self.on_next_frame_clicked)
        frame_toolbar.addWidget(self.btn_prev_frame)
        frame_toolbar.addWidget(QLabel("Chuyển:"))
        self.spin_skip_frames = QSpinBox()
        self.spin_skip_frames.setRange(1, 1000)
        self.spin_skip_frames.setValue(5)
        self.spin_skip_frames.setSuffix(" frames")
        frame_toolbar.addWidget(self.spin_skip_frames)
        frame_toolbar.addWidget(self.btn_next_frame)
        frame_toolbar.addStretch()
        self.video_frame_label = QLabel("Frame: 0 / 0")
        frame_toolbar.addWidget(self.video_frame_label)
        tab_video_layout.addLayout(frame_toolbar)

        self.video_display_label = QLabel("Chọn video để hiển thị")
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setFrameShape(QFrame.Shape.Box)
        self.video_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) # Cho phép co giãn
        tab_video_layout.addWidget(self.video_display_label, 1)

        # --- Tab 3: Grad-CAM ---
        tab_gradcam_layout = QVBoxLayout(self.tab_gradcam)
        self.gradcam_scroll_area = QScrollArea()
        self.gradcam_scroll_area.setWidgetResizable(True)
        self.gradcam_display_label = QLabel("Kết quả Grad-CAM sẽ hiển thị ở đây")
        self.gradcam_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gradcam_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.gradcam_scroll_area.setWidget(self.gradcam_display_label)
        tab_gradcam_layout.addWidget(self.gradcam_scroll_area)
        
        return display_widget

    # --- 2. QUẢN LÝ TRẠNG THÁI & UI ---

    def set_ui_for_state(self, new_state):
        self.current_state = new_state
        
        # Mặc định: Tắt
        is_idle = new_state == self.STATE_IDLE
        is_image_loaded = new_state == self.STATE_IMAGE_LOADED
        is_video_paused = new_state == self.STATE_VIDEO_PAUSED
        is_video_running = new_state == self.STATE_VIDEO_RUNNING
        is_gradcam_running = new_state == self.STATE_GRADCAM_RUNNING
        self.btn_reload_detect.setEnabled(False)
        
        # Model & Params
        self.btn_select_model.setEnabled(not (is_video_running or is_gradcam_running))
        self.params_panel.setEnabled(not (is_video_running or is_gradcam_running) and self.model is not None)
        self.combo_classes.setEnabled(not (is_video_running or is_gradcam_running) and self.model is not None)

        self.btn_reload_detect.setEnabled(is_image_loaded or is_video_paused)
        
        # Chạy / Lưu
        self.btn_run_gradcam.setEnabled((is_image_loaded or is_video_paused))
        self.btn_save_grid.setEnabled(self.current_grid_pixmap is not None and not is_gradcam_running)

        # Tab Ảnh
        self.btn_open_image.setEnabled(is_idle or is_image_loaded)
        self.btn_open_dir.setEnabled(is_idle or is_image_loaded)
        self.btn_prev_img.setEnabled(is_image_loaded and len(self.image_files_list) > 1)
        self.btn_next_img.setEnabled(is_image_loaded and len(self.image_files_list) > 1)

        # Tab Video
        self.btn_open_video.setEnabled(is_idle or is_video_paused)
        self.btn_play_pause.setEnabled(is_video_paused or is_video_running)
        self.btn_play_pause.setText("⏸️ Pause" if is_video_running else "▶️ Play")
        self.btn_stop_video.setEnabled(is_video_paused or is_video_running)
        self.video_slider.setEnabled(is_video_paused)
        self.btn_prev_frame.setEnabled(is_video_paused)
        self.btn_next_frame.setEnabled(is_video_paused)
        self.spin_skip_frames.setEnabled(is_video_paused)
        
        # Log
        if new_state == self.STATE_STARTUP:
            self.log_label.setText("Đang tải model mặc định...")
        elif new_state == self.STATE_IDLE:
            model_name = os.path.basename(self.current_model_path) if self.current_model_path else "N/A"
            self.log_label.setText(f"Model {model_name} đã tải. Sẵn sàng.")
        elif new_state == self.STATE_IMAGE_LOADED:
            img_name = os.path.basename(self.image_files_list[self.current_image_index])
            self.log_label.setText(f"Đã tải ảnh: {img_name}")
        elif new_state == self.STATE_VIDEO_PAUSED:
            self.log_label.setText(f"Video dừng. Sẵn sàng chạy Grad-CAM trên frame {self.video_current_frame_idx}.")
        elif new_state == self.STATE_VIDEO_RUNNING:
            self.log_label.setText("Đang chạy video detect...")
        elif new_state == self.STATE_GRADCAM_RUNNING:
            self.progress_bar.setValue(5)
            self.log_label.setText("Đang chạy Grad-CAM... Vui lòng đợi.")

    def update_ui_for_task(self, task_name):
        """ Khóa các tùy chọn Backward Type không tương thích """
        self.combo_backward_type.clear()
        
        if task_name == 'detect':
            self.combo_backward_type.addItems(['all', 'class', 'box'])
        elif task_name == 'segment':
            self.combo_backward_type.addItems(['all', 'class', 'box', 'segment'])
        elif task_name == 'pose':
            self.combo_backward_type.addItems(['all', 'class', 'box', 'keypoint'])
        elif task_name == 'obb':
            self.combo_backward_type.addItems(['all', 'class', 'box', 'angle'])
        elif task_name == 'classify':
            self.combo_backward_type.addItems(['all', 'class'])
        
        self.combo_backward_type.setCurrentText("all")
    
    def on_reload_detect_clicked(self):
        """
        THÊM MỚI: Chạy lại detection trên media hiện tại
        sử dụng các cấu hình mới trong panel.
        """
        try:
            current_tab = self.tabs.currentWidget()

            if current_tab == self.tab_image:
                if self.current_state == self.STATE_IMAGE_LOADED:
                    self.log_label.setText("Đang tải lại detect cho ảnh...")
                    self.run_detection_on_current_image() # Gọi lại hàm detect ảnh
                else:
                    self.show_error_message("Thông báo", "Vui lòng tải một ảnh trước.")
            
            elif current_tab == self.tab_video:
                if self.current_state == self.STATE_VIDEO_PAUSED:
                    self.log_label.setText("Đang tải lại detect cho video frame...")
                    self.run_detection_on_current_frame() # Gọi lại hàm detect video frame
                else:
                    self.show_error_message("Thông báo", "Chỉ có thể tải lại detect khi video đang dừng (Pause).")
        except Exception as e:
            self.show_error_message("Lỗi Tải Lại Detect", f"Đã xảy ra lỗi: {e}")

    # --- 3. HÀM XỬ LÝ MODEL ---
    
    def get_base_path(self):
        """ Lấy đường dẫn cơ sở (cho .py hoặc .exe) """
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def load_default_model(self):
        try:
            base_dir = self.get_base_path()
            project_root = os.path.dirname(base_dir) # lùi ra khỏi app/
            default_model_path = os.path.join(project_root, "models", "yolo11n.pt")
            if not os.path.exists(default_model_path):
                self.show_error_message("Không tìm thấy model mặc định", 
                                        f"Không thể tìm thấy 'yolo11n.pt'.\n\nVui lòng chọn model thủ công.")
                self.set_ui_for_state(self.STATE_IDLE) # Cho phép chọn
                self.set_ui_for_state(self.STATE_STARTUP) # Khóa các nút file
                return
            self.load_model_by_path(default_model_path)
        except Exception as e:
            self.show_error_message("Lỗi tải model mặc định", str(e))
            self.set_ui_for_state(self.STATE_STARTUP)

    def on_select_model_clicked(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Chọn file model YOLO", "", "Model Files (*.pt)")
        if model_path:
            self.stop_video_worker() # Dừng video nếu đang chạy
            self.load_model_by_path(model_path)
            
    def load_model_by_path(self, model_path):
        try:
            self.log_label.setText(f"Đang tải model: {os.path.basename(model_path)}...")
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.current_model_path = model_path
            
            self.combo_classes.clear()
            self.combo_classes.addItem("Tự động phát hiện", -2) 
            for index, name in self.class_names.items():
                self.combo_classes.addItem(f"{name} (ID: {index})", index)
            
            self.model_path_label.setText(self.current_model_path)
            self.model_path_label.setStyleSheet("font-style: normal; color: black;")
            self.progress_bar.setValue(100)
            
            # <<< THÊM MỚI: Cập nhật UI dựa trên task của model >>>
            task = self.model.task
            self.combo_task.setCurrentText(task)
            self.combo_task.setEnabled(False) # Khóa task
            self.update_ui_for_task(task)
            
            self.set_ui_for_state(self.STATE_IDLE)
            
        except Exception as e:
            self.model = None
            self.current_model_path = None
            self.show_error_message("Lỗi Tải Model", f"Không thể tải file model.\nLỗi: {e}\nFile có thể bị hỏng.")
            self.model_path_label.setText("Lỗi tải model.")
            self.model_path_label.setStyleSheet("font-style: italic; color: red;")
            self.progress_bar.setValue(0)
            self.set_ui_for_state(self.STATE_STARTUP)
    
    # --- 4. HÀM XỬ LÝ ẢNH TĨNH ---
    def open_image(self):
        self.stop_video_worker()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn một ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_path:
            self.image_files_list = [file_path]
            self.current_image_index = 0
            self.display_current_image()
            self.set_ui_for_state(self.STATE_IMAGE_LOADED)
            self.tabs.setCurrentWidget(self.tab_image)
            self.run_detection_on_current_image() # Tự động detect ảnh khi mở

    def open_directory(self):
        self.stop_video_worker()
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if dir_path:
            self.image_files_list = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
            for f in sorted(os.listdir(dir_path)):
                if f.lower().endswith(valid_extensions):
                    self.image_files_list.append(os.path.join(dir_path, f))
            if self.image_files_list:
                self.current_image_index = 0
                self.display_current_image()
                self.set_ui_for_state(self.STATE_IMAGE_LOADED)
                self.tabs.setCurrentWidget(self.tab_image)
                self.run_detection_on_current_image() # Tự động detect ảnh khi mở
            else:
                self.show_error_message("Không tìm thấy ảnh", "Thư mục đã chọn không chứa file ảnh hợp lệ.")
                self.set_ui_for_state(self.STATE_IDLE)

    def display_current_image(self):
        """ Hiển thị ảnh gốc (chưa detect) """
        if 0 <= self.current_image_index < len(self.image_files_list):
            path = self.image_files_list[self.current_image_index]
            pixmap = QPixmap(path)
            self.image_display_label.setPixmap(self.scale_pixmap(pixmap, self.image_display_label.size()))
            self.current_grid_pixmap = None
            self.current_image_pixmap = None # <<< THÊM MỚI: Reset pixmap đã detect
            self.btn_save_grid.setEnabled(False)

    def run_detection_on_current_image(self):
        """
        THÊM MỚI: Tự động chạy detect trên ảnh tĩnh đang được hiển thị.
        """
        if not (self.current_state == self.STATE_IMAGE_LOADED and self.model is not None):
            return
        if not (0 <= self.current_image_index < len(self.image_files_list)):
            return
            
        try:
            img_path = self.image_files_list[self.current_image_index]
            conf = self.spin_conf.value()
            imgsz = self.spin_img_size.value()
            
            # Chỉ chạy detect nếu không phải model classify
            if self.model.task == 'classify':
                self.display_current_image() # Chỉ hiển thị ảnh gốc
                return

            results = self.model(img_path, conf=conf, imgsz=imgsz, verbose=False)
            if results[0].boxes is None or len(results[0].boxes) == 0:
                self.log_label.setText("Không phát hiện đối tượng nào trong ảnh.")
                self.display_current_image() # Hiển thị ảnh gốc
                return
                
            annotated_frame = results[0].plot()
            
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            self.current_image_pixmap = pixmap # <<< LƯU LẠI pixmap đã detect
            self.image_display_label.setPixmap(self.scale_pixmap(pixmap, self.image_display_label.size()))
        
        except Exception as e:
            self.log_label.setText(f"Lỗi detect ảnh: {e}")
            self.display_current_image() # Nếu lỗi, hiển thị ảnh gốc

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            self.set_ui_for_state(self.STATE_IMAGE_LOADED)
            self.run_detection_on_current_image() # Tự động detect ảnh khi chuyển

    def next_image(self):
        if self.current_image_index < len(self.image_files_list) - 1:
            self.current_image_index += 1
            self.display_current_image()
            self.set_ui_for_state(self.STATE_IMAGE_LOADED)
            self.run_detection_on_current_image() # Tự động detect ảnh khi chuyển

    # --- 5. HÀM XỬ LÝ VIDEO ---
    def open_video(self):
        self.stop_video_worker()
        video_path, _ = QFileDialog.getOpenFileName(self, "Chọn file video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if video_path:
            try:
                self.video_capture = cv2.VideoCapture(video_path)
                if not self.video_capture.isOpened():
                    raise Exception("Không thể mở file video.")
                
                self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.video_total_frames <= 0:
                    raise Exception("Video không hợp lệ hoặc không có frame.")
                
                self.video_slider.setRange(0, self.video_total_frames - 1)
                self.current_video_path = video_path
                self.tabs.setCurrentWidget(self.tab_video)
                
                # <<< THÊM MỚI: Tự động detect frame đầu tiên >>>
                self.set_ui_for_state(self.STATE_VIDEO_PAUSED)
                self.set_video_frame(0, run_detect=True) # Tải và detect frame 0

            except Exception as e:
                self.show_error_message("Lỗi Mở Video", str(e))
                if self.video_capture: self.video_capture.release()
                self.video_capture = None
                self.set_ui_for_state(self.STATE_IDLE)

    def display_video_frame(self, frame_np):
        """ Chỉ hiển thị frame (np.ndarray) LÊN label, KHÔNG detect """
        self.current_video_frame = frame_np.copy() # Lưu frame gốc
        
        rgb_image = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_display_label.setPixmap(self.scale_pixmap(pixmap, self.video_display_label.size()))
        self.video_frame_label.setText(f"Frame: {self.video_current_frame_idx} / {self.video_total_frames}")

    def display_video_pixmap(self, pixmap):
        """ Hiển thị pixmap (đã detect) LÊN label """
        self.video_display_label.setPixmap(self.scale_pixmap(pixmap, self.video_display_label.size()))
        self.video_frame_label.setText(f"Frame: {self.video_current_frame_idx} / {self.video_total_frames}")

    def on_play_pause_clicked(self):
        if self.current_state == self.STATE_VIDEO_PAUSED:
            try:
                if self.video_worker is None:
                    conf = self.spin_conf.value()
                    imgsz = self.spin_img_size.value()
                    self.video_worker = VideoWorker(self.current_model_path, 
                                                    self.current_video_path, 
                                                    conf, imgsz)
                    self.video_worker.frame_ready.connect(self.update_video_frame)
                    self.video_worker.video_finished.connect(self.on_video_finished)
                    self.video_worker.error.connect(self.on_worker_error)
                    
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_current_frame_idx)
                    self.video_worker.cap = self.video_capture
                    self.video_worker.current_frame_idx = self.video_current_frame_idx
                    self.video_worker.start()
                else:
                    self.video_worker.resume()
                self.set_ui_for_state(self.STATE_VIDEO_RUNNING)
            except Exception as e:
                self.show_error_message("Lỗi Chạy Video", str(e))

        elif self.current_state == self.STATE_VIDEO_RUNNING:
            if self.video_worker:
                self.video_worker.pause()
            self.set_ui_for_state(self.STATE_VIDEO_PAUSED)

    def stop_video_worker(self):
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()
            self.video_worker = None
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        self.video_display_label.setText("Chọn video để hiển thị")
        self.video_frame_label.setText("Frame: 0 / 0")
        self.video_slider.setValue(0)
        self.current_video_frame = None
        self.current_video_path = None
        self.set_ui_for_state(self.STATE_IDLE)
        
    def update_video_frame(self, pixmap, frame_np, frame_idx):
        if self.current_state == self.STATE_VIDEO_RUNNING:
            self.display_video_pixmap(pixmap)
            self.current_video_frame = frame_np
            self.video_current_frame_idx = frame_idx
            self.video_slider.setValue(frame_idx)

    def on_video_finished(self):
        self.log_label.setText("Video kết thúc.")
        self.stop_video_worker()
        
    def set_video_frame(self, frame_index, run_detect=False):
        """ Tua video đến frame, hiển thị frame GỐC, và (tùy chọn) chạy detect """
        if self.video_capture and self.current_state == self.STATE_VIDEO_PAUSED:
            frame_index = max(0, min(frame_index, self.video_total_frames - 1))
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_capture.read()
            if ret:
                self.video_current_frame_idx = frame_index
                self.video_slider.setValue(frame_index)
                self.display_video_frame(frame) # Hiển thị frame gốc NGAY LẬP TỨC
                
                if run_detect:
                    # Chạy detect (debounce) sau 10ms
                    self.detect_timer.start(10) 
            
    def run_detection_on_current_frame(self):
        """ Chạy detect trên self.current_video_frame và cập nhật label """
        if self.current_video_frame is not None and self.current_state == self.STATE_VIDEO_PAUSED:
            try:
                conf = self.spin_conf.value()
                imgsz = self.spin_img_size.value()
                results = self.model(self.current_video_frame, conf=conf, imgsz=imgsz, verbose=False)
                annotated_frame = results[0].plot()
                
                # Chuyển đổi và hiển thị
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.display_video_pixmap(pixmap) # Cập nhật lại với frame đã detect
            except Exception as e:
                self.log_label.setText(f"Lỗi detect frame: {e}")

    # --- Xử lý Slider Video và Nút Frame ---
    def on_slider_pressed(self):
        self.slider_is_pressed = True
        
    def on_slider_released(self):
        self.slider_is_pressed = False
        if self.current_state == self.STATE_VIDEO_PAUSED:
            new_index = self.video_slider.value()
            self.set_video_frame(new_index, run_detect=True) # Tua và detect
            
    def on_prev_frame_clicked(self):
        skip = self.spin_skip_frames.value()
        self.set_video_frame(self.video_current_frame_idx - skip, run_detect=True)

    def on_next_frame_clicked(self):
        skip = self.spin_skip_frames.value()
        self.set_video_frame(self.video_current_frame_idx + skip, run_detect=True)

    # --- 6. HÀM XỬ LÝ GRAD-CAM ---
    def run_gradcam(self):
        # 1. Kiểm tra trạng thái
        if self.current_state not in [self.STATE_IMAGE_LOADED, self.STATE_VIDEO_PAUSED]:
            self.show_error_message("Không thể chạy", "Chỉ có thể chạy Grad-CAM trên ảnh tĩnh hoặc video đang dừng.")
            return

        # 2. Lấy nguồn ảnh (File hoặc Frame)
        img_source = None
        source_for_detect = None # Nguồn để chạy auto-detect
        
        if self.current_state == self.STATE_IMAGE_LOADED:
            if 0 <= self.current_image_index < len(self.image_files_list):
                img_source = self.image_files_list[self.current_image_index]
                source_for_detect = img_source
        
        elif self.current_state == self.STATE_VIDEO_PAUSED:
            if self.current_video_frame is not None:
                img_source = self.current_video_frame.copy()
                # Lưu frame tạm ra đĩa để model YOLO đọc (dễ xử lý hơn)
                try:
                    cv2.imwrite(self.temp_frame_path, img_source)
                    source_for_detect = self.temp_frame_path
                except Exception as e:
                    self.show_error_message("Lỗi Lưu Frame Tạm", f"Không thể lưu frame video tạm: {e}")
                    return
        
        if img_source is None:
            self.show_error_message("Lỗi Nguồn Ảnh", "Không tìm thấy ảnh hoặc frame video hợp lệ.")
            return
            
        # 3. Lấy lớp (Class)
        selected_class_index = self.combo_classes.currentData()
        selected_class_name = self.combo_classes.currentText()
        
        if selected_class_index == -2: # Tự động phát hiện
            self.log_label.setText("Đang tự động phát hiện lớp...")
            QApplication.processEvents()
            try:
                results = self.model(source_for_detect, verbose=False, conf=self.spin_conf.value())
                task = self.model.task

                if task == 'classify':
                    if results[0].probs is None: raise Exception("Model Classify không trả về 'probs'.")
                    selected_class_index = int(results[0].probs.top1)
                    conf = float(results[0].probs.top1conf)
                    selected_class_name = f"{self.class_names[selected_class_index]} (Conf: {conf:.2f})"
                else: # detect, segment, pose, obb
                    if results[0].boxes is None or len(results[0].boxes) == 0:
                        raise Exception("Không phát hiện đối tượng nào.")
                    top_result = results[0].boxes[0]
                    selected_class_index = int(top_result.cls[0])
                    conf = float(top_result.conf[0])
                    selected_class_name = f"{self.class_names[selected_class_index]} (Conf: {conf:.2f})"
                
                self.log_label.setText(f"Đã phát hiện: {selected_class_name}. Bắt đầu chạy...")

            except Exception as e:
                self.show_error_message("Tự động phát hiện thất bại", f"Không thể tự động phát hiện lớp.\nLỗi: {e}")
                self.set_ui_for_state(self.previous_state if self.previous_state else self.STATE_IDLE)
                return
        
        # 4. Thu thập các tham số khác và Xử lý lỗi
        try:
            layers_str = self.edit_layers.text()
            if not layers_str.strip():
                 raise ValueError("Ô 'Layers' không được để trống.")
            layers_list = [int(l.strip()) for l in layers_str.replace(' ', '').split(',') if l.strip()]
            if not layers_list:
                raise ValueError("Danh sách layer không hợp lệ.")
            
            params_dict = {
                'weight': self.current_model_path,
                'device': self.combo_device.currentText(),
                'method': self.combo_method.currentText(),
                'layer': layers_list,
                'backward_type': self.combo_backward_type.currentText(),
                'conf_threshold': self.spin_conf.value(),
                'ratio': self.spin_ratio.value(),
                'renormalize': self.check_renormalize.isChecked(),
                'task': self.combo_task.currentText(),
                'img_size': self.spin_img_size.value(),
                'show_result': self.check_show_boxes.isChecked()
            }
        except ValueError as e:
            self.show_error_message("Lỗi Tham Số", f"Giá trị nhập không hợp lệ.\nLỗi: {e}\nKiểm tra ô 'Layers' (ví dụ: 19, 22).")
            return
        
        # 5. Chạy Worker
        self.previous_state = self.current_state
        self.set_ui_for_state(self.STATE_GRADCAM_RUNNING)
        
        self.gradcam_worker = GradCamWorker(img_source, 
                                          selected_class_index, 
                                          selected_class_name, 
                                          self.current_model_path,
                                          params_dict)
        self.gradcam_worker.finished.connect(self.on_gradcam_finished)
        self.gradcam_worker.progress.connect(self.progress_bar.setValue)
        self.gradcam_worker.log_message.connect(self.log_label.setText)
        self.gradcam_worker.error.connect(self.on_worker_error) # Kết nối tín hiệu lỗi
        self.gradcam_worker.start()

    def on_gradcam_finished(self, grid_pixmap):
        self.progress_bar.setValue(100)
        self.current_grid_pixmap = grid_pixmap
        self.gradcam_display_label.setPixmap(self.scale_pixmap(grid_pixmap, self.gradcam_scroll_area.size()))
        self.tabs.setCurrentWidget(self.tab_gradcam)
        self.set_ui_for_state(self.previous_state)

    def save_grid(self):
        if self.current_grid_pixmap:
            save_path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh Grid", "gradcam_grid.png", "PNG Image (*.png)")
            if save_path:
                try:
                    self.current_grid_pixmap.save(save_path, "PNG")
                    self.log_label.setText(f"Đã lưu ảnh grid tại: {save_path}")
                except Exception as e:
                    self.show_error_message("Lỗi Lưu Ảnh", str(e))
        else:
            self.show_error_message("Lỗi Lưu Ảnh", "Không có ảnh grid để lưu.")

    # --- 7. HÀM TIỆN ÍCH (HELPER) ---
    
    def on_tab_changed(self, index):
        """ Cập nhật trạng thái khi đổi tab """
        if self.current_state in [self.STATE_GRADCAM_RUNNING, self.STATE_VIDEO_RUNNING]:
             return # Không làm gì nếu đang chạy
             
        if index == 0: # Tab Ảnh
            if self.image_files_list:
                self.set_ui_for_state(self.STATE_IMAGE_LOADED)
            else:
                self.set_ui_for_state(self.STATE_IDLE)
        elif index == 1: # Tab Video
            if self.current_video_path:
                self.set_ui_for_state(self.STATE_VIDEO_PAUSED)
            else:
                self.set_ui_for_state(self.STATE_IDLE)
        # Tab Grad-CAM không đổi trạng thái
        
    def on_worker_error(self, title, message):
        """ Slot nhận tín hiệu lỗi từ bất kỳ worker nào """
        self.show_error_message(title, message)
        # Quay lại trạng thái trước khi chạy
        self.set_ui_for_state(self.previous_state) 
        
    def show_error_message(self, title, message):
        self.log_label.setText(f"LỖI: {message.splitlines()[0]}")
        QMessageBox.critical(self, title, message)

    def scale_pixmap(self, pixmap, size):
        """ Co giãn pixmap để vừa với label mà giữ tỷ lệ """
        return pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def resizeEvent(self, event):
        """ Tự động co giãn ảnh khi resize cửa sổ """
        super().resizeEvent(event)
        
        # Lấy kích thước của label cha
        image_label_size = self.image_display_label.size()
        video_label_size = self.video_display_label.size()
        gradcam_label_size = self.gradcam_scroll_area.viewport().size() # Kích thước của viewport
        
        if self.tabs.currentWidget() == self.tab_image:
            if self.current_image_pixmap: # <<< THAY ĐỔI: Ưu tiên ảnh đã detect
                self.image_display_label.setPixmap(self.scale_pixmap(self.current_image_pixmap, image_label_size))
            elif self.image_files_list:
                self.display_current_image() # Quay về ảnh gốc nếu chưa detect

        elif self.tabs.currentWidget() == self.tab_video and self.current_video_frame is not None:
            # Khi resize video, chỉ hiển thị frame gốc cho mượt
            # (Detect sẽ chạy lại khi người dùng dừng kéo)
            self.display_video_frame(self.current_video_frame)

        elif self.tabs.currentWidget() == self.tab_gradcam and self.current_grid_pixmap is not None:
            self.gradcam_display_label.setPixmap(self.scale_pixmap(self.current_grid_pixmap, gradcam_label_size))

    def closeEvent(self, event):
        self.stop_video_worker()
        # Dọn dẹp thư mục tạm
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Không thể xóa thư mục tạm: {e}")
        event.accept()
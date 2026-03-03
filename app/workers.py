import os
import cv2
import sys
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage
from ultralytics import YOLO
from gradcam_logic import yolo_heatmap # Import logic từ file kia

class GradCamWorker(QThread):
    finished = Signal(QPixmap) 
    progress = Signal(int)
    log_message = Signal(str)
    error = Signal(str, str) # Tín hiệu lỗi: (title, message)

    def __init__(self, img_source, class_index, class_name, model_path, params_dict):
        super().__init__()
        self.img_source = img_source 
        self.target_class_index = class_index
        self.target_class_name = class_name
        self.model_path = model_path
        self.params_dict = params_dict
        self.temp_dir = "temp_gradcam_output"
        os.makedirs(self.temp_dir, exist_ok=True)

    def run(self):
        try:
            if isinstance(self.img_source, str):
                self.log_message.emit(f"Bắt đầu xử lý ảnh: {os.path.basename(self.img_source)}")
            else:
                self.log_message.emit(f"Bắt đầu xử lý frame video...")
            self.progress.emit(10)

            layers_to_run = self.params_dict['layer'] 
            base_params = self.params_dict.copy()
            total_steps = len(layers_to_run) + 2
            heatmap_paths = []
            
            # 1. Chạy cho từng layer riêng lẻ
            for i, l in enumerate(layers_to_run):
                layer_idx = l
                self.log_message.emit(f"Đang chạy layer {layer_idx} cho lớp '{self.target_class_name}'...")
                current_params = base_params.copy()
                current_params['layer'] = [layer_idx]
                
                model = yolo_heatmap(**current_params, target_class=self.target_class_index)
                save_path = os.path.join(self.temp_dir, f'cls_{self.target_class_index}_layer{layer_idx:02d}.png')
                
                model(self.img_source, save_path)
                heatmap_paths.append((f"Layer {layer_idx}", save_path))
                self.progress.emit(int(10 + (i + 1) / total_steps * 80))

            # 2. Chạy cho 'mean'
            self.log_message.emit(f"Đang chạy 'mean' cho lớp '{self.target_class_name}'...")
            mean_params = base_params.copy()
            mean_params['layer'] = layers_to_run
            
            model_mean = yolo_heatmap(**mean_params, target_class=self.target_class_index)
            mean_save_path = os.path.join(self.temp_dir, f'mean_cls_{self.target_class_index}.png')
            model_mean(self.img_source, mean_save_path)
            heatmap_paths.append(("Mean Layer", mean_save_path))
            self.progress.emit(90)

            # 3. Tạo ảnh Grid
            self.log_message.emit("Đang ghép ảnh kết quả (grid)...")
            
            if isinstance(self.img_source, str):
                img_orig_cv = cv2.imread(self.img_source)
                if img_orig_cv is None: raise Exception(f"Không thể đọc lại ảnh gốc: {self.img_source}")
            else:
                img_orig_cv = self.img_source
            
            grid_image = self.create_grid_image(img_orig_cv, heatmap_paths)
            
            # 4. Chuyển đổi sang QPixmap
            # Kiểm tra xem grid_image có phải BGR không
            if len(grid_image.shape) < 3: # Nếu là ảnh xám
                q_image = QImage(grid_image.data, grid_image.shape[1], grid_image.shape[0], grid_image.strides[0], QImage.Format.Format_Grayscale8)
            else:
                grid_image_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(grid_image_rgb.data, grid_image.shape[1], grid_image.shape[0], grid_image.strides[0], QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            
            self.progress.emit(100)
            self.log_message.emit("Hoàn tất!")
            self.finished.emit(pixmap)

        except Exception as e:
            import traceback
            error_msg = f"Lỗi Grad-CAM: {e}\n\nChi tiết:\n{traceback.format_exc()}"
            self.error.emit("Lỗi Chạy Grad-CAM", error_msg)
            self.progress.emit(0)

    # (create_grid_image và resize_and_add_title giữ nguyên)
    def create_grid_image(self, img_orig_cv, heatmap_paths):
        images_with_titles = []
        img_orig = self.resize_and_add_title(img_orig_cv, "Original Image")
        images_with_titles.append(img_orig)
        
        for title, path in heatmap_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = self.resize_and_add_title(img, title)
                    images_with_titles.append(img)
        
        num_images = len(images_with_titles)
        if num_images <= 1: # Nếu chỉ có ảnh gốc (lỗi)
             return images_with_titles[0]
        
        num_cols = 2
        num_rows = (num_images + num_cols - 1) // num_cols
        
        # Đảm bảo tất cả ảnh cùng kích thước (nếu không sẽ lỗi)
        target_h, target_w = images_with_titles[0].shape[:2]
        
        grid = np.full((target_h * num_rows, target_w * num_cols, 3), 255, dtype=np.uint8)
        
        for i, img in enumerate(images_with_titles):
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h)) # Resize lần nữa nếu bị lệch
            row = i // num_cols
            col = i % num_cols
            grid[row*target_h : (row+1)*target_h, col*target_w : (col+1)*target_w] = img
        return grid

    def resize_and_add_title(self, img, title, target_size=(480, 480)):
        if img is None: return np.full((target_size[1] + 30, target_size[0], 3), 255, dtype=np.uint8)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        h, w, ch = img_resized.shape if len(img.shape) > 2 else (img.shape[0], img.shape[1], 1)
        
        if ch == 1: # Chuyển ảnh xám sang BGR
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        final_img = np.full((h + 30, w, 3), 255, dtype=np.uint8)
        final_img[30:h+30, :] = img_resized
        cv2.putText(final_img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return final_img

# Worker chạy video detect (khác với Grad-CAM)
class VideoWorker(QThread):
    frame_ready = Signal(QPixmap, np.ndarray, int) # (pixmap đã vẽ, frame gốc, frame_idx)
    video_finished = Signal()
    error = Signal(str, str)

    def __init__(self, model_path, video_path, conf_threshold, img_size):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model = None
        self.cap = None
        self._is_running = True
        self._is_paused = False
        self.current_frame_idx = 0

    def run(self):
        try:
            self.model = YOLO(self.model_path)
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở video: {self.video_path}")
            
            while self._is_running:
                while self._is_paused:
                    if not self._is_running: break
                    self.msleep(100)
                if not self._is_running:
                    break

                ret, frame = self.cap.read()
                if not ret:
                    break # Hết video
                
                results = self.model(frame, conf=self.conf_threshold, imgsz=self.img_size, verbose=False)
                annotated_frame = results[0].plot()
                
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                self.frame_ready.emit(pixmap, frame.copy(), self.current_frame_idx)
                self.current_frame_idx += 1

        except Exception as e:
            self.error.emit("Lỗi Video", f"Lỗi khi xử lý video: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.video_finished.emit()

    def stop(self):
        self._is_running = False
        self._is_paused = False

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False
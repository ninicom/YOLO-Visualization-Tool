# YOLO Grad-CAM Visualizer 🚀🔥

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-black)
![PySide6](https://img.shields.io/badge/PySide6-GUI-green)

A comprehensive, interactive Graphical User Interface (GUI) tool for visualizing and explaining YOLO models using various Class Activation Mapping (CAM) methods. Built with **PySide6** and **Ultralytics**, this tool helps researchers and computer vision engineers understand what their models are looking at, enabling deeper analysis of architectural improvements and custom heads.

## ✨ Key Features

* **Multi-Task Support:** Seamlessly works with all YOLO tasks including **Detect, Segment, Pose, OBB (Oriented Bounding Boxes), and Classify**.
* **Interactive Media Handling:** * Analyze static images or entire directories.
  * Load videos, play/pause, and run Grad-CAM on specific frames dynamically.
* **10+ Visualization Methods:** Supports `GradCAM`, `GradCAMPlusPlus`, `XGradCAM`, `EigenCAM`, `HiResCAM`, `LayerCAM`, and more.
* **Deep Architectural Analysis:** Select specific layers (e.g., specific blocks or heads) to extract feature maps and visualize their individual or mean activations.
* **Asynchronous Processing:** Utilizes `QThread` workers for heavy model inference and video processing, ensuring a smooth, non-blocking UI experience.
* **Auto-Class Detection:** Automatically detects the highest-confidence class in a frame to target for CAM generation.
* **Clean & Modular Architecture:** Structured following industry standards with robust cross-platform path handling and isolated business logic.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ninicom/yolo-gradcam-visualizer.git
   cd yolo-gradcam-visualizer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Since the core application logic is isolated in the `app/` directory, run the main script from the root of the project:

```bash
python app/main.py
```

### Quick Start Guide:
1. **Load Model:** The tool automatically loads `models/default_model.pt` on startup. You can click `Chọn file Model (.pt) khác` to load your custom weights.
2. **Configure Parameters:** Select the XAI method, target layers, device (CPU/CUDA), and confidence thresholds in the configuration panel.
3. **Select Media:** Navigate to the `🖼️ Ảnh Tĩnh` (Image) or `🎬 Video` tab to load your media.
4. **Run Grad-CAM:** Pause at the desired frame (if using video) and click `🔥 Chạy Grad-CAM`. The tool will process the selected layers and display a grid comparing the original image with the generated heatmaps.
5. **Save:** Save the generated grid directly to your machine.

## 📂 Project Structure

```text
YOLO-GradCAM-Visualizer/
├── app/                     # Core application source code
│   ├── main.py              # Application entry point
│   ├── ui_mainwindow.py     # GUI layout and event handling
│   ├── workers.py           # QThread implementation for async tasks
│   └── gradcam_logic.py     # XAI core logic and YOLO model wrapper
├── assets/                  # Images and media for documentation
│   └── demo_screenshot.png  
├── models/                  # Directory for storing model weights
│   └── default_model.pt     # Default fallback model
├── .gitignore               # Ignored files and directories
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ninicom/yolo-gradcam-visualizer/issues).

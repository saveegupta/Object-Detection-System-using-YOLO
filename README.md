# Vehicle Detection using YOLOv8

## Project Overview
This repository contains an implementation of vehicle detection models using different variants of YOLOv8. The project aims to compare the performance of YOLOv8n, YOLOv8m, and YOLOv8x models for detecting vehicles on roads.

## Dataset
The project uses a vehicle detection dataset specifically formatted for YOLOv8 training. The dataset contains images of vehicles moving on roads, with appropriate bounding box annotations.

## Models Compared
Three different YOLOv8 model variants were trained and evaluated:
- YOLOv8n (Nano): The smallest and fastest model, suitable for edge devices
- YOLOv8m (Medium): A balanced model offering good trade-off between speed and accuracy
- YOLOv8x (XLarge): The largest and most accurate model, requiring more computational resources

## Implementation Details
- **Framework**: Ultralytics YOLO
- **Training**: Each model was trained for 30 epochs with 640×640 image size and batch size of 8
- **Hardware**: Models were trained using GPU acceleration (when available)
- **Evaluation Metrics**: mAP50, mAP50-95, Precision, and Recall

## Results
The repository includes comprehensive evaluation results for each model:
- Performance metrics comparison
- Confusion matrices
- Visualization of results

## Requirements
- Python 3.x
- PyTorch
- Ultralytics YOLO
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Usage

### Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/vehicle-detection-yolov8.git
cd vehicle-detection-yolov8

# Install dependencies
pip install -r requirements.txt
```

### Training
```python
from ultralytics import YOLO
import torch

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = YOLO("yolov8m.pt").to(device)

# Train model
model.train(data="path/to/data.yaml", epochs=30, imgsz=640, batch=8)
```

### Evaluation
```python
# Validate the model
metrics = model.val()

# Print results
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p.mean()}")
print(f"Recall: {metrics.box.r.mean()}")
```

## Project Structure
```
vehicle-detection-yolov8/
├── data.yaml              # Dataset configuration
├── models/                # Trained model weights
├── notebooks/            
│   └── model_training.ipynb  # Training notebook
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Future Work
- Implement real-time vehicle detection
- Optimize models for edge deployment
- Extend to multi-class vehicle classification
- Add vehicle tracking functionality

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to Ultralytics for developing YOLOv8
- Credit to the creators of the vehicle detection dataset

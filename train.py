from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m-cls.pt")

# Dataset Path
Data_set = 'Dataset'

# Save Model Path
Train_Result = 'Train_Result'

# Model Train
result = model.train(
    data='Dataset',
    epochs=100,
    optimizer='AdamW',
    project='Train_Result',
    device='mps',
    imgsz='224',
)
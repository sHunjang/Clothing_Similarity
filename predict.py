from ultralytics import YOLO

# Load Pretrained Model
Pretrained_Model = YOLO('Train_Result/train6/weights/best.pt')

# Test Clothing Image Path
Predict_Images_Path = 'test_set'

# Save Predict Result Path
Predict_Result_Path = 'Predict_Result'

# Predict Model
result_combination = Pretrained_Model.predict(source=Predict_Images_Path, save=True, save_txt=True, project=Predict_Result_Path, device='mps')
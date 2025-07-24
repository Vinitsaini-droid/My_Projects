# infer.py
from ultralytics import YOLO
import os

# Load model
model = YOLO("model/best.pt")

# Run prediction on test images
results = model.predict(
    source='G:\My Drive\LunarVisionSubmissions\test_images',  # folder with test images
    save_txt=True,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    project='runs/predict',
    name='test_preds',
    device=0
)

# Copy results to predictions folder
import shutil
import os

pred_dir = 'runs/predict/test_preds/labels'
os.makedirs('predictions', exist_ok=True)
for file in os.listdir(pred_dir):
    shutil.copy(os.path.join(pred_dir, file), 'predictions')

# train.py
from ultralytics import YOLO

# Load model 
model = YOLO('G:\My Drive\LunarVisionSubmissions\model')

# Train on your custom dataset
model.train(
    data='G:\My Drive\LunarVisionSubmissions\data.yaml',
    epochs=50,
    imgsz=640,
    save=True,
    save_period=1,
    project='runs/detect',
    name='lunar_crater_boulder',
    batch=16,
    device=0,  
    augment=True
)

# Save best.pt to model folder manually
import shutil
shutil.copy('runs/detect/lunar_crater_boulder/weights/best.pt', '../model/best.pt')

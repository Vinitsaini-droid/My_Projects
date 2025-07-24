# Install required packages 
#!pip install -q ultralytics gradio

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

from torch.nn import functional as F
from matplotlib import cm

# === Load YOLOv8 model ===
model = YOLO("/content/drive/MyDrive/LunarVisionSubmissions/model/best.pt")
model.eval()

# === Hook to capture last feature map ===
feature_maps = []
def hook_fn(module, input, output):
    feature_maps.append(output)

target_layer = model.model.model[-2]  # Last conv layer
hook_handle = target_layer.register_forward_hook(hook_fn)

def generate_gradcam(image_path):
    feature_maps.clear()

    # === Load & preprocess image ===
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_norm = img_resized.astype(np.float32) / 255.0

    # === Predict with YOLO ===
    result = model.predict(image_path, conf=0.25)[0]
    if len(result.boxes) == 0:
        return img_rgb, img_rgb  # No detections

    # === Use top box ===
    top_box = result.boxes[0]
    cls_id = int(top_box.cls.item())
    class_name = model.names[cls_id]

    # === Get Grad-CAM ===
    fmap = feature_maps[0].squeeze(0)  # (C, H, W)
    weights = top_box.conf.item()

    cam = fmap.mean(dim=0).cpu().detach().numpy()
    cam = cv2.resize(cam, (640, 640))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize

    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    return img_resized, overlay

# === Gradio UI ===
gr.Interface(
    fn=generate_gradcam,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[
        gr.Image(label="Original"),
        gr.Image(label="Grad-CAM")
    ],
    title="Grad-CAM for YOLOv8 Crater/Boulder Detector"
).launch(share=True)

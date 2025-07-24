import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np

def load_image(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    return raw_image, transform(raw_image).unsqueeze(0)  # shape: (1, 3, 640, 640)

def apply_gradcam(image_path, model_path, output_path="gradcam_result.jpg"):
    # ✅ Load model and set eval mode
    model = YOLO(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ✅ Load image
    raw_image, image_tensor = load_image(image_path)
    image_tensor = image_tensor.to(device)

    # ✅ Hook into YOLO backbone layer
    fmap = None
    def forward_hook(module, input, output):
        nonlocal fmap
        fmap = output

    # Register hook to 2nd last layer in backbone
    handle = model.model.model[-2].register_forward_hook(forward_hook)

    # ✅ Run a dummy prediction to trigger the hook
    _ = model(image_tensor)

    # ✅ Remove hook
    handle.remove()

    # ✅ Convert feature map to heatmap
    fmap = fmap.squeeze(0)             # remove batch dim: (C, H, W)
    cam = fmap.mean(0).detach().cpu()  # average across channels -> (H, W)
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize to [0,1]
    cam_img = transforms.ToPILImage()(cam.unsqueeze(0))

    # ✅ Resize and overlay
    cam_img = cam_img.resize(raw_image.size)
    overlay = Image.blend(raw_image, cam_img.convert("RGB").resize(raw_image.size), alpha=0.5)

    # ✅ Save and show
    overlay.save(output_path)
    print(f"✅ Grad-CAM saved to: {output_path}")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

# === Run ===
if __name__ == "__main__":
    apply_gradcam(
        image_path="gradcam_input.jpg",      
        model_path="../model/best.pt",            
        output_path="gradcam_resnet.jpg"                   
    )

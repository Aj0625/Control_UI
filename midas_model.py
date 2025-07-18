import cv2
import numpy as np
import torch

def run_midas(input_path, output_path, colormap_path, model=None, transform=None, device=None):
    """
    Runs MiDaS depth estimation on an input image.

    Args:
        input_path (str): Path to input image.
        output_path (str): Grayscale depth output path.
        colormap_path (str): Depth overlay color image output path.
        model (torch.nn.Module): Preloaded MiDaS model (optional).
        transform (function): Associated transform for the model (optional).
        device (torch.device): Torch device (optional).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Failed to read image from {input_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load model and transform if not passed (fallback mode)
    if model is None or transform is None:
        print("[INFO] Loading MiDaS model dynamically (fallback)...")
        model_type = "DPT_Large"
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        model.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

    # Transform and run inference
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path, depth_norm)

    # Overlay color depth on RGB
    colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, colormap, 0.4, 0)
    cv2.imwrite(colormap_path, overlay)

    print(f"[INFO] MiDaS depth estimation complete for {input_path}")

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import ConvNextImageProcessor, ConvNextForImageClassification

def visualize_heatmap(model, processor, image_path, device):
    """
    使用 Grad-CAM 對單張圖片生成熱力圖
    """
    orig_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    target_layer = model.convnext.encoder.stages[-1].layers[-1].dwconv
    
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=None)
        grayscale_cam = grayscale_cam[0, :]
    
    rgb_img = np.float32(image_rgb) / 255.0
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")
    plt.show()

if __name__ == "__main__":
    best_output_dir = "/home/jovyan/course-img-classification/checkpoints/convnext-large-finetuned_best"
    model = ConvNextForImageClassification.from_pretrained(best_output_dir)
    processor = ConvNextImageProcessor.from_pretrained(best_output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_image_path = "/home/jovyan/course-img-classification/dataset/test/unknown/test_0001.jpg"  
    visualize_heatmap(model, processor, test_image_path, device)

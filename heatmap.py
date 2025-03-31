import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import matplotlib
matplotlib.use('Agg')  # 在導入 pyplot 之前設置後端
import matplotlib.pyplot as plt

def visualize_heatmap(model, processor, image_path, device, ori_lable, predict_lable, output_dir="heatmap_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    try:
        orig_image = cv2.imread(image_path)
        if orig_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = image_rgb.shape[:2]
        
        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        original_forward = model.forward
        
        try:
            def new_forward(input_tensor=None, pixel_values=None):
                tensor_to_use = pixel_values if pixel_values is not None else input_tensor
                outputs = original_forward(pixel_values=tensor_to_use)
                return outputs.logits
            
            model.forward = new_forward
            
            target_layer = model.convnext.encoder.stages[-1].layers[-1].dwconv
            
            from pytorch_grad_cam import ScoreCAM
            with ScoreCAM(model=model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=None)
                grayscale_cam = grayscale_cam[0, :]
        finally:
            model.forward = original_forward
        
        grayscale_cam_resized = cv2.resize(grayscale_cam, (original_width, original_height))
        
        rgb_img = np.float32(image_rgb) / 255.0
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f"Original Image: {ori_lable}")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Score-CAM Heatmap: {predict_lable}")
        plt.axis("off")
        
        plt.tight_layout()
        
        image_name = os.path.basename(image_path)
        
        output_file = os.path.join(output_dir, f"cam_{image_name.split('.')[0]}.png")
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close('all') 
        
        return output_file
    
    except Exception as e:
        print(f"生成熱力圖時出錯 ({image_path}): {str(e)}")
        return ""

# def visualize_heatmap(model, processor, image_path, device, output_dir="heatmap_results"):
#     os.makedirs(output_dir, exist_ok=True)
    
#     model.eval()
    
#     orig_image = cv2.imread(image_path)
#     if orig_image is None:
#         raise ValueError(f"Could not read image: {image_path}")
    
#     image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
#     original_height, original_width = image_rgb.shape[:2]
    
#     inputs = processor(images=image_rgb, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     original_forward = model.forward
#     inputs["pixel_values"].requires_grad = True
    
#     try:
#         def new_forward(input_tensor=None, pixel_values=None):
#             tensor_to_use = pixel_values if pixel_values is not None else input_tensor
#             outputs = original_forward(pixel_values=tensor_to_use)
#             return outputs.logits
        
#         model.forward = new_forward
        
#         target_layer = model.convnext.encoder.stages[-1].layers[-1].dwconv
        
#         with GradCAM(model=model, target_layers=[target_layer]) as cam:
#             grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=None)
#             grayscale_cam = grayscale_cam[0, :]
#     finally:
#         model.forward = original_forward
    
#     grayscale_cam_resized = cv2.resize(grayscale_cam, (original_width, original_height))
    
#     rgb_img = np.float32(image_rgb) / 255.0
    
#     visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)
    
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_rgb)
#     plt.title("Original Image")
#     plt.axis("off")
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(visualization)
#     plt.title("Grad-CAM Heatmap")
#     plt.axis("off")
    
#     plt.tight_layout()
    
#     image_name = os.path.basename(image_path)
    
#     output_file = os.path.join(output_dir, f"gradcam_{image_name}")
    
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return output_file

if __name__ == "__main__":
    best_output_dir = "/home/nckusoc/桌面/claire/course-img-classification/checkpoints/convnext-large-finetuned_best"
    model = ConvNextForImageClassification.from_pretrained(best_output_dir)
    processor = ConvNextImageProcessor.from_pretrained(best_output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_image_path = "/home/nckusoc/桌面/claire/course-img-classification/dataset/test/unknown/test_0001.jpg"  
    
    heatmap_path = visualize_heatmap(model, processor, test_image_path, device)
    print(f"Heatmap saved to: {heatmap_path}")
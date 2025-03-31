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
    # 設置模型為評估模式
    model.eval()
    
    # 讀取並轉換圖片 (cv2 預設為 BGR，需要轉為 RGB)
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError(f"無法讀取圖片: {image_path}")
    
    image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    # 使用 processor 預處理圖片
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 進行前向傳播，獲取預測結果
    with torch.no_grad():
        outputs = model(inputs["pixel_values"])
    
    # 獲取預測的類別
    predicted_class = outputs.logits.argmax(-1).item()
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_prob = probs[0, predicted_class].item()
    
    # 如果模型有類別標籤，則獲取類別名稱
    class_names = getattr(model.config, "id2label", None)
    if class_names:
        predicted_label = class_names[predicted_class]
        print(f"預測類別: {predicted_label} (類別 ID: {predicted_class})")
    else:
        print(f"預測類別 ID: {predicted_class}")
    
    print(f"預測概率: {predicted_prob:.4f}")
    
    # 選擇目標層：使用最後一個 stage 中最後一層的 dwconv 層
    try:
        target_layer = model.convnext.encoder.stages[-1].layers[-1].dwconv
    except AttributeError:
        print("模型結構與預期不符，嘗試查找正確的目標層...")
        # 檢查模型結構
        if hasattr(model, 'convnext'):
            print("模型有 convnext 屬性")
            if hasattr(model.convnext, 'encoder'):
                print("模型有 convnext.encoder 屬性")
                if hasattr(model.convnext.encoder, 'stages') and len(model.convnext.encoder.stages) > 0:
                    print(f"模型有 {len(model.convnext.encoder.stages)} 個 stages")
                    last_stage = model.convnext.encoder.stages[-1]
                    if hasattr(last_stage, 'layers') and len(last_stage.layers) > 0:
                        print(f"最後一個 stage 有 {len(last_stage.layers)} 個 layers")
                        last_layer = last_stage.layers[-1]
                        print(f"最後一層的屬性: {dir(last_layer)}")
                    else:
                        print("最後一個 stage 沒有 layers 屬性或為空")
                else:
                    print("模型沒有 stages 屬性或為空")
            else:
                print("模型沒有 convnext.encoder 屬性")
        else:
            print("模型沒有 convnext 屬性")
            
        # 打印完整模型結構
        print("\n完整模型結構:")
        print(model)
        
        raise ValueError("無法找到目標層，請根據打印的模型結構手動指定正確的目標層路徑")
    
    # 建立 GradCAM 並生成熱力圖
    with GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type=="cuda") as cam:
        # 創建一個目標函數，而不是直接使用整數類別
        target_category_function = lambda x: x[:, predicted_class]
        
        # 對預測類別生成熱力圖
        grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=[target_category_function])
        grayscale_cam = grayscale_cam[0, :]
    
    # 將原圖正規化到 [0,1]
    rgb_img = np.float32(image_rgb) / 255.0
    
    # 生成熱力圖疊加的圖片
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 顯示結果：原圖和熱力圖並排
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    if class_names:
        plt.title(f"原始圖片\n預測: {predicted_label} ({predicted_prob:.2%})")
    else:
        plt.title(f"原始圖片\n預測類別 ID: {predicted_class} ({predicted_prob:.2%})")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM 熱力圖")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"gradcam_result_{image_path.split('/')[-1]}", dpi=300, bbox_inches='tight')
    plt.show()
    
    return visualization

if __name__ == "__main__":
    model_dir = "/Users/apple/Downloads/2025/NLP-image-classification/checkpoints/convnext-large-finetuned_best"
    
    try:
        model = ConvNextForImageClassification.from_pretrained(model_dir)
        processor = ConvNextImageProcessor.from_pretrained(model_dir)
        print(f"成功加載模型和處理器從: {model_dir}")
    except Exception as e:
        print(f"加載模型時出錯: {e}")
        raise
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    model.to(device)
    
    test_image_path = "/Users/apple/Downloads/2025/NLP-image-classification/dataset/test/unknown/test_0001.jpg"
    print(f"處理圖片: {test_image_path}")
    visualize_heatmap(model, processor, test_image_path, device)
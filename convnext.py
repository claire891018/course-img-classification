import torch
import yaml
import os
import random
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from torchvision.transforms import RandomErasing
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from heatmap import visualize_heatmap

with open('configs/convnext_ft_cfg.yaml', "r") as f:
    config = yaml.safe_load(f)

class ImageTextDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_training=True):
        """
        transform (callable, optional): Optional transform to be applied on a sample.
        is_training: 是否為訓練集，決定使用哪種資料處理方式
        """
        data = pd.read_csv(csv_path)
        self.annotations = data['img_path']
        self.labels = data['img_label']
        self.mapping = {'spaghetti': 0, 'ramen': 1, 'udon': 2}
        self.is_training = is_training
        
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.Resize((420, 420)),
                    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomGrayscale(p=0.02),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.2),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.mapping[self.labels.iloc[idx]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": images}, torch.tensor(labels, dtype=torch.long)

def train(model, train_loader, optimizer, criterion, scheduler, device, gradient_clip_val=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        images, labels = batch
        images = images["pixel_values"].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return total_loss / len(train_loader), accuracy

def evaluate(model, test_loader, criterion, device, epoch, processor=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    results = []

    generate_heatmaps = (epoch >= 6)
    heatmap_dir = f"heatmap_results/{config['output_dir']}/epoch_{epoch}"

    if generate_heatmaps and processor is not None:
        os.makedirs(heatmap_dir, exist_ok=True)

    test_paths = [test_loader.dataset.annotations.iloc[i] for i in range(len(test_loader.dataset))]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = batch
            images_tensor = images["pixel_values"].to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=images_tensor)
            loss = criterion(outputs.logits, labels)
            probabilities = F.softmax(outputs.logits, dim=1)
            
            total_loss += loss.item()
            
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                actual_idx = batch_idx * test_loader.batch_size + i
                if actual_idx >= len(test_paths):
                    continue  
                
                img_path = test_paths[actual_idx]

                heatmap_path = ""
                if generate_heatmaps and processor is not None:
                    heatmap_type = ""
                    if predicted[i].item() != labels[i].item():
                        if probabilities[i][predicted[i]].item() > 0.9:
                            heatmap_type = "high_confidence_errors"
                        else:
                            heatmap_type = "incorrect_predictions"
                            

                    if heatmap_type:
                        type_dir = os.path.join(heatmap_dir, heatmap_type)
                        os.makedirs(type_dir, exist_ok=True)
                        
                        try:
                            heatmap_path = visualize_heatmap(
                                model,
                                processor,
                                img_path,
                                device,
                                ori_lable=model.config.id2label[labels[i].item()],
                                predict_lable=model.config.id2label[predicted[i].item()],
                                output_dir=type_dir,
                            )
                        except Exception as e:
                            print(f"生成熱力圖時出錯 ({img_path}): {e}")
                    
                results.append({
                    "img_path": img_path,
                    "true_label": labels[i].item(),
                    "predicted_label": predicted[i].item(),
                    "class_name_true": model.config.id2label[labels[i].item()],
                    "class_name_pred": model.config.id2label[predicted[i].item()],
                    "probability_spaghetti": probabilities[i][0].item(),
                    "probability_ramen": probabilities[i][1].item(),
                    "probability_udon": probabilities[i][2].item(),
                    "correct": int(predicted[i].item() == labels[i].item()),
                    "loss": loss.item(),
                    "heatmap_path": heatmap_path
                })
                    
    result_dir = f"results/{config['output_dir']}" 
    os.makedirs(result_dir, exist_ok=True)  

    csv_filename = f"{result_dir}/{epoch}.csv"        
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)

    json_filename = f"{result_dir}/{epoch}_for_llm.json"
    df.to_json(json_filename, orient="records", indent=2)

    accuracy = 100.0 * correct / total
    return total_loss / len(test_loader), accuracy

# def evaluate(model, test_loader, criterion, device, epoch):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     results = []
#     sample_idx = 0

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating"):
#             images, labels = batch
#             images = images["pixel_values"].to(device)
#             labels = labels.to(device)
            
#             outputs = model(pixel_values=images)
#             loss = criterion(outputs.logits, labels)
#             probabilities = F.softmax(outputs.logits, dim=1)
            
#             total_loss += loss.item()
            
#             _, predicted = outputs.logits.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#             for i in range(labels.size(0)):
#                 results.append({
#                     "sample_idx": f"test_{sample_idx:04d}.jpg",
#                     "true_label": labels[i].item(),
#                     "predicted_label": predicted[i].item(),
#                     "probabilities": probabilities[i].cpu().numpy(),  
#                     "correct": int(predicted[i].item() == labels[i].item()),
#                     "loss": loss.item()
#                 })
#                 sample_idx += 1
                    
#     csv_filename = f"dataset/results_{epoch}.csv"          
#     df = pd.DataFrame(results)
#     df.to_csv(csv_filename, index=False)

#     accuracy = 100.0 * correct / total
#     return total_loss / len(test_loader), accuracy

def main():
    print(f"Starting training with config: {config}")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("PyTorch sees {} GPUs.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = ImageTextDataset(config['train_csv'], is_training=True)
    test_dataset = ImageTextDataset(config['test_csv'], is_training=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model_name = config['model_name']
    
    processor = ConvNextImageProcessor.from_pretrained(model_name)
    model = ConvNextForImageClassification.from_pretrained(model_name)
    
    num_labels = len(config['class_names'])
    model.config.num_labels = num_labels
    model.config.id2label = {i: class_name for i, class_name in enumerate(config['class_names'])}
    model.config.label2id = {class_name: i for i, class_name in enumerate(config['class_names'])}
    
    classifier_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(classifier_features, num_labels)
    
    # 可選：凍結部分層以進行更有效的微調 (特別適用於小數據集)
    if config.get('freeze_base_model', False):
        print("Freezing ConvNeXt base layers...")
        # 凍結特徵提取器的層，只訓練分類頭
        for name, param in model.convnext.named_parameters():
            param.requires_grad = False
    
    model.to(device)
    
    params = [
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n], 
            "lr": float(config['encoder_lr'])},
        {"params": model.classifier.parameters(), 
            "lr": float(config['classifier_lr'])}
    ]
    
    optimizer = torch.optim.AdamW(
        params, 
        weight_decay=float(config['weight_decay'])
    )
    
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(config['label_smoothing_factor'])
    )
    
    total_steps = len(train_loader) * config['num_train_epochs']
    warmup_steps = int(total_steps * float(config['warmup_ratio']))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    writer = SummaryWriter(log_dir=f"logs/{config['output_dir']}")
    
    best_accuracy = 0.0
    patience = config.get('patience', 3)
    patience_counter = 0
    
    print(f"Starting training for {config['num_train_epochs']} epochs...")
    for epoch in range(config['num_train_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_train_epochs']}")
        
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, scheduler, device,
            gradient_clip_val=config.get('gradient_clip_val', 1.0)
        )
        
        # test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, epoch)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, epoch, processor)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        output_dir = f"checkpoints/{config['output_dir']}/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            
            best_output_dir = f"checkpoints/{config['output_dir']}_best"
            os.makedirs(best_output_dir, exist_ok=True)
            model.save_pretrained(best_output_dir)
            processor.save_pretrained(best_output_dir)
            print(f'保存最佳模型，準確率: {best_accuracy:.2f}%')
        else:
            patience_counter += 1
            print(f"準確率未提升，耐心計數器: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f'提前停止於epoch {epoch+1}，最佳準確率: {best_accuracy:.2f}%')
                break
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    writer.close()

if __name__ == "__main__":
    main()
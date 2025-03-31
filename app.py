from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import json
import os
import random
import pandas as pd
import uvicorn
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware



load_dotenv(override=True)

app = FastAPI(title="Image Classification Analysis API", 
              description="使用 Gemma 3 IT 模型分析圖像分類結果和熱力圖")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None

class AnalysisRequest(BaseModel):
    result_json_path: str
    num_samples: int = 15
    output_dir: str = "gemma_analysis"

class AnalysisResponse(BaseModel):
    task_id: str
    message: str

class AnalysisResult(BaseModel):
    sample_id: str
    true_label: str
    predicted_label: str
    correct: bool
    confidence: float
    analysis: str


class ChatMessage(BaseModel):
    role: str  # "system" or "user" or "assistant"
    content: str

class ChatImageRequest(BaseModel):
    """同時帶有圖片資訊與多輪對話的結構"""
    image_data: Dict[str, Any]  # 放 img_path, class_name_true, class_name_pred, probability_xxx 等
    messages: List[ChatMessage] 
    
def load_model():
    global model, processor
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
    
    model_id = "google/gemma-3-4b-it"
    
    if model is None:
        print("Loading Gemma 3 IT model...")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        print(f"Model loaded on {model.device}")
    
    if processor is None:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        print("Processor loaded")

def analyze_sample(sample, output_dir):
    global model, processor
    
    if model is None or processor is None:
        load_model()
    
    if not os.path.exists(sample["heatmap_path"]):
        return None
    
    try:
        image = Image.open(sample["heatmap_path"])
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert in image classification analysis. Your task is to analyze classification results and attention maps (heatmaps)."}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"""
Analyze this image classification result and heatmap:

Image Classification Details:
- True label: {sample['class_name_true']}
- Predicted label: {sample['class_name_pred']}
- Prediction correct: {'Yes' if sample['correct'] else 'No'}
- Prediction probabilities: 
  * Spaghetti: {sample['probability_spaghetti']:.4f}
  * Ramen: {sample['probability_ramen']:.4f}
  * Udon: {sample['probability_udon']:.4f}

The image shows:
- Left side: Original image
- Right side: Attention heatmap (red areas indicate regions the model focused on)

Please analyze:
1. How did the model make this decision?
2. Are the highlighted areas reasonable?
3. Are the features the model focused on relevant to the food type?
4. If the prediction is wrong, what might be the reason?
5. What are possible improvements?
                    """}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
            generation = generation[0][input_len:]
        
        analysis = processor.decode(generation, skip_special_tokens=True)
        
        sample_id = os.path.basename(sample["img_path"]).split('.')[0]
        # output_file = os.path.join(output_dir, f"analysis_{sample_id}.txt")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     f.write(analysis)
            
        return {
            "sample_id": sample_id,
            "true_label": sample["class_name_true"],
            "predicted_label": sample["class_name_pred"],
            "correct": sample["correct"],
            "confidence": max(sample["probability_spaghetti"], sample["probability_ramen"], sample["probability_udon"]),
            "analysis": analysis,
            # "analysis_path": output_file,
            "heatmap_path": sample["heatmap_path"],
            "image_path": sample["img_path"]
        }
    
    except Exception as e:
        print(f"Error analyzing sample {sample['img_path']}: {str(e)}")
        return None

def analyze_results_task(result_json_path: str, num_samples: int, output_dir: str, task_id: str):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(result_json_path, 'r') as f:
        results = json.load(f)
    
    status_file = os.path.join(output_dir, f"task_{task_id}_status.json")
    with open(status_file, 'w') as f:
        json.dump({"status": "running", "progress": 0, "total": 0}, f)
    
    high_conf_errors = [r for r in results if r["predicted_label"] != r["true_label"] and 
                        r["probability_" + r["class_name_pred"]] > 0.9 and r["heatmap_path"]]
    
    general_errors = [r for r in results if r["predicted_label"] != r["true_label"] and 
                     r["probability_" + r["class_name_pred"]] <= 0.9 and r["heatmap_path"]]
    
    correct_samples = [r for r in results if r["predicted_label"] == r["true_label"] and r["heatmap_path"]]
    
    all_samples = []
    if high_conf_errors:
        print(f"Found {len(high_conf_errors)} high confidence errors")
        all_samples.extend(random.sample(high_conf_errors, min(num_samples, len(high_conf_errors))))
    if general_errors:
        print(f"Found {len(general_errors)} general errors")
        all_samples.extend(random.sample(general_errors, min(num_samples, len(general_errors))))
    if correct_samples:
        print(f"Found {len(correct_samples)} correct samples")
        all_samples.extend(random.sample(correct_samples, min(num_samples, len(correct_samples))))
    
    with open(status_file, 'w') as f:
        json.dump({"status": "running", "progress": 0, "total": len(all_samples)}, f)
    
    analysis_results = []
    for i, sample in enumerate(all_samples):
        print(f"Analyzing sample {i+1}/{len(all_samples)}: {sample['img_path']}")
        result = analyze_sample(sample, output_dir)
        if result:
            analysis_results.append(result)
        
        with open(status_file, 'w') as f:
            json.dump({"status": "running", "progress": i+1, "total": len(all_samples)}, f)
    
    summary_df = pd.DataFrame(analysis_results)
    summary_df.to_csv(os.path.join(output_dir, f"analysis_summary_{task_id}.csv"), index=False)
    
    with open(status_file, 'w') as f:
        json.dump({"status": "completed", "progress": len(all_samples), "total": len(all_samples)}, f)
    
    return analysis_results

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_images(request: AnalysisRequest, background_tasks: BackgroundTasks):
    task_id = f"task_{random.randint(10000, 99999)}"
    
    background_tasks.add_task(
        analyze_results_task, 
        request.result_json_path, 
        request.num_samples, 
        request.output_dir,
        task_id
    )
    
    return {"task_id": task_id, "message": "Analysis task started"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str, output_dir: str = "gemma_analysis"):
    status_file = os.path.join(output_dir, f"task_{task_id}_status.json")
    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Task not found")
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    return status

@app.get("/results/{task_id}")
async def get_results(task_id: str, output_dir: str = "gemma_analysis"):
    summary_file = os.path.join(output_dir, f"analysis_summary_{task_id}.csv")
    if not os.path.exists(summary_file):
        raise HTTPException(status_code=404, detail="Results not found")
    
    df = pd.read_csv(summary_file)
    return df.to_dict(orient="records")

@app.post("/chat-image")
async def chat_image(request: ChatImageRequest):
    """
    使用 Gemma 3 IT 模型，結合給定的圖片資訊 (image_data) 和當前對話 (messages)，
    進行多輪對話。
    """
    global model, processor
    if model is None or processor is None:
        load_model()
        
    image_path = request.image_data.get('img_path', None)
    heatmap_path = request.image_data.get('heatmap_path', None)
    
    try:
        text_context = f"""
        請分析這張麵條圖片的分類結果：
        - 真實標籤: {request.image_data.get('class_name_true', '未知')}
        - 模型預測標籤: {request.image_data.get('class_name_pred', '未知')}
        - 預測是否正確: {'是' if request.image_data.get('correct', 0) == 1 else '否'}
        - 分類概率：
          義大利麵: {request.image_data.get('probability_spaghetti', 0.0):.4f}
          拉麵:     {request.image_data.get('probability_ramen', 0.0):.4f}
          烏龍麵:   {request.image_data.get('probability_udon', 0.0):.4f}
        
        請分析模型為什麼會得出這樣的分類結果？請從圖片特徵和模型判斷角度分析。
        """
        
        messages = []
        
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": "你是一個專業的麵條圖像分類分析助手。你可以根據圖片和分類數據分析模型的判斷理由。"}]
        })
        
        user_content = []
        
        if image_path and os.path.exists(image_path):
            try:
                original_image = Image.open(image_path)
                user_content.append({"type": "image", "image": original_image})
                print(f"成功加載原始圖片: {image_path}")
            except Exception as e:
                print(f"無法加載原始圖片 {image_path}: {str(e)}")
        
        user_content.append({"type": "text", "text": text_context.strip()})
        
        if not request.image_data.get('correct', True) and heatmap_path and os.path.exists(heatmap_path):
            try:
                heatmap_image = Image.open(heatmap_path)
                user_content.append({"type": "image", "image": heatmap_image})
                user_content.append({"type": "text", "text": "這是模型的熱力圖，顯示了模型關注的區域（紅色越深表示關注度越高）。"})
                print(f"成功加載熱力圖: {heatmap_path}")
            except Exception as e:
                print(f"無法加載熱力圖 {heatmap_path}: {str(e)}")
        
        if request.messages:
            user_query = request.messages[0].content 
            query_item = {"type": "text", "text": user_query}
            combined_content = user_content.copy() 
            combined_content.insert(0, query_item)  
            messages.append({
                "role": "user",
                "content": combined_content
            })
        else:
            messages.append({
                "role": "user",
                "content": user_content
            })
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=1024,  # 增加生成長度
                do_sample=True
            )
            generation = generation[0][input_len:]
        
        response_text = processor.decode(generation, skip_special_tokens=True)
        
        return {"response": response_text}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"處理過程中出錯: {str(e)}\n{error_details}")
        return {"response": f"抱歉，處理請求時出錯: {str(e)}。詳細錯誤: {error_details[:200]}..."}
    
@app.post("/reload-env")
async def reload_environment():
    load_dotenv(override=True)
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
    
    return {"status": "environment reloaded"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
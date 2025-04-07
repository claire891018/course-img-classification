# NoodleNeXt

本專案使用 **ConvNeXt** 模型進行圖像分類 (麵類辨識)，並透過 **ScoreCAM** 生成熱力圖，協助分析錯誤樣本與可解釋性。部分功能與 **LLM** 整合，提供多輪對話式分析。
[This project's Github Repo](https://github.com/claire891018/course-img-classification.git)

NCKU NLP course midterm. [ref](https://github.com/naoya1110/ai_robotics_lab_2024_hands_on.git)

## 目錄結構

```
.
├── app.py                    # 主要訓練 & 評估程式入口（含 ConvNeXt 微調與熱力圖生成）
├── checkpoints/              # 訓練過程中儲存的模型檔案
│   ├── convnext-large-finetuned_best
│   └── ... (其他檔)
├── configs/                  # 訓練配置 YAML
│   └── convnext_ft_cfg.yaml
├── convnext.py               # 模型相關程式碼 (若有)
├── dataset/
│   ├── train/                # 訓練圖片資料
│   ├── test/                 # 測試圖片資料
│   ├── results_{epoch}.csv   # 每輪測試預測結果
│   ├── results_{epoch}_for_llm.json  # 每輪測試預測結果 JSON（整合 LLM）
│   └── ... (其他 CSV 檔)
├── demo/
│   ├── index.html            # 前端頁面（展示圖片、熱力圖、與LLM互動等）
│   ├── scripts.js
│   └── styles.css
├── heatmap.py                # 生成熱力圖的主程式或函式 (若為獨立)
├── heatmap_results/
│   ├── epoch_1/              # 存放第1輪測試時產生的熱力圖
│   ├── epoch_2/
│   └── ...
├── logs/                     # TensorBoard 或其他日誌輸出
├── requirements.txt          # Python 套件需求
└── README.md                 # 本檔
```

## 安裝環境

1. 建議使用 Python 3.8+
2. 建議使用虛擬環境 (e.g., `venv`)
3. 安裝依賴套件
   ```bash
   pip install -r requirements.txt
   ```

## 使用方式

1. **編輯配置檔**

   - 在 `configs/convnext_ft_cfg.yaml` 中調整以下參數：
     - `train_csv` / `test_csv`：指定 CSV 檔路徑（其中包含 `img_path`, `img_label`）
     - `class_names`：模型欲辨識的類別
     - `model_name`：底層ConvNeXt模型
     - `num_train_epochs`, `batch_size`, `encoder_lr`, `classifier_lr` 等訓練參數
2. **執行訓練**

   ```bash
   python app.py
   ```

   - 依照配置檔內容，程式將進行微調訓練並在每個 epoch 評估測試集
   - 完成後會輸出
     - `checkpoints/`：存放中間與最佳模型權重
     - `dataset/results_{epoch}.csv` / `dataset/results_{epoch}_for_llm.json`：存放預測結果
     - `heatmap_results/epoch_{epoch}/`：若有錯誤樣本，產生對應的熱力圖
3. **檢視日誌 (TensorBoard)**

   - 進入專案根目錄或 logs 目錄：
     ```bash
     tensorboard --logdir=logs/
     ```
   - 在瀏覽器打開 `http://localhost:6006` 查看訓練和測試的 Loss / Accuracy
4. **查看熱力圖**

   - 於 `heatmap_results/epoch_{epoch}/` 內查看對應的圖片與熱力圖檔案（分為高信心錯誤 `high_confidence_errors` 及一般錯誤 `incorrect_predictions`）。

## 其他說明

- 若要與前端整合，請參考 `demo/` 資料夾內的靜態頁面（index.html、scripts.js、styles.css）。透過後端API可讀取 `results_{epoch}.csv` 或 `results_{epoch}_for_llm.json`，在網頁上展示圖片與預測結果、可視化熱力圖。
- 若要與 LLM 進行對話式分析，可在 `app.py` 或其他檔案中擴充 FastAPI 路由，將 CSV/JSON 結果傳給 LLM，並於前端以 `fetch` 方式呼叫後端 API，實作多輪對話。

---

如需更詳盡的技術或設計細節，可參考 `demo/document.md`

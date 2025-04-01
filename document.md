# NoodleNeXt：ConvNeXt 麵類影像分類與 LLM 可解釋性分析

本專案以 ConvNeXt 為基底模型，針對「麵類食物」進行圖像分類，並結合 ScoreCAM 生成熱力圖，以解析模型對錯誤樣本的判斷依據。並與 LLM 整合，進行多輪對話式分析與解釋。

## 1. 模型選擇

在本專案的初期實驗中，我測試兩類視覺模型：**Vision Transformer (ViT)** 與 **ConvNeXt**。下文從模型概念、優勢與挑戰三個方面進行比較，並說明最終選定 ConvNeXt 作為主要影像分類與特徵抽取骨幹的理由。

| 模型                     | 全局上下文捕捉 | 硬體部署效率 | 資料需求量高 |
| ------------------------ | -------------- | ------------ | ------------ |
| Vision Transformer (ViT) | ✔             | ✘           | ✔           |
| ConvNeXt                 | ✘             | ✔           | ✘           |

### 1.1 Vision Transformer (ViT)

#### 模型概念

ViT [1] 將 Transformer 架構（最初應用於自然語言處理 [2]）直接拓展到圖像領域，其核心流程包括：

- **影像拆分與向量化**：將輸入影像拆分為固定大小（如 16×16）的 Patch，並透過線性映射轉換成向量表示，同時加入位置編碼以保留空間資訊。
- **多頭自注意力**：利用多頭自注意力機制與前饋神經網路（FFN）堆疊多層 Transformer Encoder，最終通過額外的分類 Token 或線性分類器得到影像分類結果。

#### 模型優勢與挑戰

- **👍 全局上下文捕捉**：全域自注意力能收集影像內部的全局資料，並且注意力權重可視化提供一定的可解釋性。
- **👎 資料依賴性**：ViT 在資料量較小或標籤較少的情況下（幾千筆至萬筆資料），需要更強的正則化與資料增強策略，或融合卷積前端以彌補其歸納偏差不足的問題。

### 1.2 ConvNeXt

#### 模型概念

ConvNeXt [4] 採取「卷積網路的再現代化 (Modernizing a ConvNet)」策略，將近年 Transformer 等領域的設計元素融入傳統 CNN 架構。其主要設計包括：

- **大尺寸卷積核**：採用 7×7 或更大尺寸的卷積核以擴大感受野。
- **Depthwise Convolution 與 Inverted Bottleneck**

  * **Depthwise Convolution：** 對每個輸入通道分別進行卷積，降低參數數量與計算量。例如，對於一個 3×3 的卷積核，若輸入有 64 個通道，傳統卷積需要 *64×3×3×輸出通道數* 的參數，而深度卷積僅需 *64×3×3* 個參數。
  * **Inverted Bottleneck：** 與傳統瓶頸結構不同（寬→窄→寬），倒置瓶頸先擴展通道再進行深度卷積處理，最後再縮減通道數（窄→寬→窄。）。這種設計有助於捕捉高階特徵，同時保持較低的計算成本。
- **LayerNorm 替代 BatchNorm**：使用 LayerNorm 簡化實作並避免依賴 batch 大小。

#### 模型優勢與挑戰

- **👍 可擴充性與高效率**：當網路深度與寬度提升或資料規模增加，ConvNeXt 能夠在模型規模（參數量、層數、通道數）增加時，有效利用更多的計算資源和數據，呈現出良好 Scaling 效果，而無需額外設計如相對位置編碼或窗口注意力。
- **👎 局部感受野限制**：雖然卷積固有的歸納偏差對中小型資料集有幫助，但在捕捉全局上下文方面可能不如自注意力靈活。

### 1.3 整合與最終選擇

綜合我的實驗結果以及考量下列因素，我選擇 **ConvNeXt** 作為本專案的主要骨幹架構：

1. **資料規模與模型穩定性**：本專案的訓練資料不屬於超大規模（超過百萬張圖像），ConvNeXt 的局部歸納偏差使其在中等資料量下更易於穩定收斂。
2. **差異化嘗試**：由於同學已實作 ViT，因此嘗試不同架構的模型以提供多元觀點。

<table>
  <tr>
    <th>模型</th>
    <th>圖像尺寸</th>
    <th>測試準確率 (%)</th>
    <th>Epoch 數</th>
  </tr>
  <tr style="background-color: #8b33dc;">
    <td>ConvNeXt-large</td>
    <td>384×384</td>
    <td>97.66</td>
    <td>8</td>
  </tr>
  <tr style="background-color: #ef1b90;">
    <td>ConvNeXt-base</td>
    <td>224×224</td>
    <td>96.91</td>
    <td>11</td>
  </tr>
  <tr style="background-color: #26bbcc;">
    <td>ViT-large (patch32, lr5e-5)</td>
    <td>384×384</td>
    <td>96.51</td>
    <td>9</td>
  </tr>
  <tr style="background-color: #ffb403;">
    <td>ViT-base (patch32, lr5e-5)</td>
    <td>224x224</td>
    <td>96.05</td>
    <td>9</td>
  </tr>
  <tr style="background-color: #21bccd;">
    <td>ViT-base (patch32, lr1e-4)</td>
    <td>224x224</td>
    <td>96.05</td>
    <td>9</td>
  </tr>
</table>

![tensorboard](https://imgur.com/gVMyj7i.jpg)

## 2. 資料集、資料增強與訓練策略

### 2.1 資料集說明

- 本專案的數據包含三種麵類：
  - 義大利麵（spaghetti）：300 張
  - 拉麵（ramen）：600 張
  - 烏龍麵（udon）：1000 張
- 訓練集總計 1900 張圖片，而測試集總計 4500 張圖片。

### 2.2 資料增強策略

我採用多種資料增強技術，增強數據多樣性並降低過擬合風險：

- **尺寸調整與裁剪**：將圖片 Resize 到固定尺寸後，再以 RandomResizedCrop 隨機裁剪至模型輸入大小（384×384）。
- **翻轉與旋轉**：以 50% 機率水平翻轉，並隨機旋轉（±10°）。
- **顏色變化**：透過 ColorJitter 調整亮度、對比度與飽和度，並偶爾將圖片轉為灰階。
- **仿射變換**：使用 RandomAffine 進行平移等變換。
- **隨機遮蔽**：利用 RandomErasing 隨機抹除部分區域，迫使模型學習更具魯棒性的特徵。

這些增強方法能讓模型見到不同變化的圖像，進而提升泛化能力。

```python
transforms.Compose([
    transforms.Resize((420, 420)),                         # 調整大小略大於目標尺寸
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),  # 隨機裁剪
    transforms.RandomHorizontalFlip(p=0.5),                # 水平翻轉
    transforms.RandomRotation(10),                         # 隨機旋轉
    transforms.ColorJitter(brightness=0.2, contrast=0.2,   # 顏色增強
                          saturation=0.2),
    transforms.RandomGrayscale(p=0.02),                    # 隨機灰度化
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 隨機平移
    transforms.ToTensor(),                                 # 轉換為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # 標準化
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),                       # 隨機擦除
])
```

### 2.3 訓練策略

- **超參數設置**：
  - 批次大小設定為 16，總訓練 epoch 為 15。
  - Weight Decay 設為 0.03 與 Label Smoothing Factor 0.2，以**控制模型複雜度並降低過擬合**。
- **學習率與優化器**：
  - 整體學習率為 2e-5，分類頭與主幹分別使用 1e-4 與 1e-5 的學習率，反映預訓練模型微調時對不同層級進行精細調整。
  - 採用 Cosine Learning Rate Decay 策略，並在訓練初期使用 15% 的 Warmup 階段，幫助模型穩定收斂。
- **穩定性調整**：
  - 使用 Gradient Clipping（上限 1.0）來防止梯度爆炸，確保每次參數更新不會過大。
  - 設定 Early Stopping（Patience=10），若驗證指標連續 10 個 epoch 未提升則提前停止訓練，以避免過擬合。

## 3. 模型總體性能

### 3.1 整體性能

在第 8 + 8 個 epoch，模型達到最佳準確率。

* 測試集準確率：97.73%
* 平均損失值：0.518 （損失值看起來較高，是因為模型訓練時使用 Label Smoothing（0.2）使!得損失值提高。）

### 3.2 分類別性能

![matrix](https://imgur.com/CxEw8ft.jpg)

| 類別                 | 準確率 |
| -------------------- | ------ |
| 義大利麵 (spaghetti) | 97.8%  |
| 拉麵 (ramen)         | 98.0%  |
| 烏龍麵 (udon)        | 97.4%  |

> 📍烏龍麵準確度最低，很容易與拉麵

## 4. 可解釋性分析

### 4.1 ScoreCAM 模型可視化解釋

我利用 ScoreCAM 技術來揭示模型在影像分類任務中的決策依據，並展示模型在處理圖像時對哪些區域更為敏感。

#### 技術原理

**現有方法介紹：** 在視覺模型解釋中，GradCAM 與 ScoreCAM 是兩種主要方法：

- **GradCAM** 通過反向傳播計算目標類別在特徵圖上的梯度，進而對每個激活通道（即卷積層輸出中各獨立的 feature map）進行加權，生成一張反映圖像中各區域對最終預測貢獻度的熱力圖。
- **ScoreCAM** 將目標層中每個激活通道的激活圖上採樣至與原始圖像相同的尺寸，再分別作為遮罩應用到原始圖像上。接著，ScoreCAM 透過前向傳播計算每個遮罩對預測分數的影響，得到一個權重分數。最後，將各通道的遮罩按照權重進行加權求和，生成一張綜合熱力圖。

**為何選擇 ScoreCAM？**

![GradCAM](https://i.imgur.com/IpS4A1J.jpg)
![ScoreCAM](https://i.imgur.com/vvwlB08.jpg)

- **不依賴梯度：** 由於 ScoreCAM 依靠前向激活與預測分數來評估每個激活通道，避免 GradCAM 在多層正規化和非線性激活中可能出現的梯度不穩定問題。
- **目標層選擇：** 我選擇 ConvNeXt 最後一層的 depthwise 卷積作為目標層。此處的 depthwise 卷積在每個輸入通道上獨立運算，能夠捕捉較高階特徵。

#### 實作方式

在實際實作中，我採取如下策略來應用 ScoreCAM 生成熱力圖，以解釋模型決策過程：

1. **圖像預處理與設備轉移：** 將原始圖像讀入、轉換為 RGB 格式，再通過 ConvNeXt 專用的圖像處理器進行預處理，最終將處理後的數據移至 GPU 上進行運算。
2. **修改模型 Forward 方法：** 為了讓 ScoreCAM 直接利用模型最終的預測分數，我暫時覆蓋了模型的 forward 函數，使其僅返回最終的 logits（分類得分），而忽略中間層的輸出。這樣，ScoreCAM 能夠在前向傳播中僅根據最終結果來對激活圖進行打分。
3. **目標層的選取：** 我選擇 ConvNeXt 模型中最後一個階段的最後一層 depthwise 卷積作為目標層。該層的激活圖保留了豐富的高階特徵資訊，同時具備較高的空間分辨率，能夠精確反映模型對圖像中局部區域的關注程度。
4. **ScoreCAM 熱力圖生成流程：**

   - 將目標層中每個激活通道的激活圖上採樣至與原始圖像相同的尺寸。
   - 對每個激活通道，利用上採樣後的激活圖作為遮罩應用到原始圖像上，並通過前向傳播計算該遮罩對預測分數的影響，以獲得每個通道的權重。
   - 最後，將所有激活通道根據其權重進行加權求和，得到一張綜合的灰階熱力圖，該熱力圖直觀地顯示了模型在最終預測中重點關注的圖像區域。

總結來說，我通過暫時修改模型 forward 方法來專注於最終預測分數，並選取 ConvNeXt 的最後一層 depthwise 卷積作為目標層，進而利用 ScoreCAM 生成反映模型注意力分佈的熱力圖。這種方法能夠有效揭示模型決策過程中的關鍵區域，並且相較於 GradCAM 更加穩定與高效。

### 4.2 LLM 輔助解釋工具

為了進一步提升模型的可解釋性，除了 ScoreCAM 熱力圖外，我還整合了大型語言模型 (LLM) 作為輔助解釋工具。具體使用預訓練模型 `google/gemma-3-4b-it`，將 ConvNeXt 模型的預測結果、ScoreCAM 熱力圖及相關數據轉換為易懂的自然語言說明。其核心流程如下：

1. **結構化數據整合**：首先，我從 ConvNeXt 模型中獲取預測結果（例如預測類別、信心分數）以及通過 ScoreCAM 生成的熱力圖。這些資訊被整理為一個結構化的輸入，包括：

   - 模型對圖像的最終預測與相應信心分數；
   - 真實標籤；
   - 熱力圖顯示的模型注意力區域。
2. **LLM 解釋生成**：使用預訓練模型 google/gemma-3-4b-it 作為語言生成器，將上述結構化數據輸入進去。LLM 會根據這些資料生成一段自然語言解釋，描述模型在圖像中關注的關鍵區域，以及這些區域如何影響最終預測。
3. 以網頁形式與 LLM 互動。

   ![ ](https://imgur.com/tFJcojj.jpg "網頁入口")

   ![img](https://imgur.com/e4kfdoH.jpg "LLM ans example1")

> 結合 `google/gemma-3-4b-it`，我能夠將 ConvNeXt 模型輸出與 ScoreCAM 熱力圖的技術數據轉換成自然語言解釋。 ScoreCAM 提高模型可解釋性的透明度，LLM 使錯誤案例的原因分析和後續改進說明更潛顯易懂。

## 參考文獻

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021.
2. Vaswani, A., Shazeer, N., Parmar, N., et al. "Attention is All You Need." *NeurIPS*, 2017.
3. Liu, Z., Lin, Y., Cao, Y., et al. "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *ICCV*, 2021, pp. 10012–10022.
4. Liu, Z., Mao, H., Wu, C.-Y., et al. "A ConvNet for the 2020s." *CVPR*, 2022, pp. 11976–11986.

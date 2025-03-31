import streamlit as st
import pandas as pd
import os
import random
from PIL import Image
import requests
from io import BytesIO
import json

st.set_page_config(
    page_title="NoodleNeXt - 麵條分類模型",
    page_icon="🍜",
    layout="wide"
)

avatars = ['Alexander', 'Valentina', 'Aiden', 'Leah', 'Brooklynn', 'Kingston', 'Jade', 'Jack', 'Kimberly']

@st.cache_data
def load_data():
    data = pd.read_csv("/home/nckusoc/桌面/claire/course-img-classification/dataset/results_7.csv")
    return data

def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        st.error(f"無法加載圖片 {image_path}: {e}")
        return None

def get_avatar(seed):
    url = f"https://api.dicebear.com/9.x/thumbs/svg?seed={seed}"
    return url

def project_details():
    st.title("NoodleNeXt: ConvNeXt 麵類影像分類")
    # 修改點 2: 使用 with open 讀取 markdown 文件
    with open("/home/nckusoc/桌面/claire/course-img-classification/document.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    with st.expander("", expanded=True):
        st.markdown(md_content)

def results_page():
    st.title("分類結果")
    
    data = load_data()
    
    st.sidebar.header("篩選")
    result_type = st.sidebar.multiselect(
        "預測結果",
        options=["正確", "錯誤"],
        default=["正確", "錯誤"]
    )
    
    class_filter = st.sidebar.multiselect(
        "麵條類型",
        options=["spaghetti", "ramen", "udon"],
        default=["spaghetti", "ramen", "udon"]
    )
    
    filtered_data = data.copy()
    if "正確" in result_type and "錯誤" not in result_type:
        filtered_data = filtered_data[filtered_data["correct"] == 1]
    elif "錯誤" in result_type and "正確" not in result_type:
        filtered_data = filtered_data[filtered_data["correct"] == 0]
    
    if len(class_filter) < 3:
        filtered_data = filtered_data[filtered_data["class_name_true"].isin(class_filter)]
    
    items_per_page = 9
    num_pages = max(1, (len(filtered_data) + items_per_page - 1) // items_per_page)
    page = st.sidebar.number_input("頁數", min_value=1, max_value=num_pages, value=1) - 1
    
    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_data))
    page_data = filtered_data.iloc[start_idx:end_idx]
    
    st.write(f"顯示 {start_idx+1}-{end_idx} 筆，共 {len(filtered_data)} 筆")
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(page_data.iterrows()):
        col = cols[i % 3]
        
        with col:
            avatar = random.choice(avatars)
            avatar_url = get_avatar(avatar)
            
            with st.container():
                st.markdown(f"<img src='{avatar_url}' width='40' style='border-radius:50%;'/>", unsafe_allow_html=True)
                
                img_path = row["img_path"]
                img = load_image(img_path)
                if img:
                    # 修改點 3: 將 use_column_width 替換為 use_container_width
                    st.image(img, use_container_width=True)
                
                correct = row["correct"] == 1
                color = "green" if correct else "red"
                st.markdown(f"<span style='color:{color};font-weight:bold;'>{'✓ 正確' if correct else '✗ 錯誤'}</span>", unsafe_allow_html=True)
                st.write(f"真實: {row['class_name_true']}")
                st.write(f"預測: {row['class_name_pred']} ({row['probability_' + row['class_name_pred'].lower()]*100:.1f}%)")
                
                # 修改點 4: 檢查 CSV 中的 heatmap_path 是否為空值，且只有錯誤預測才會有值
                if not correct and row["heatmap_path"] and os.path.exists(row["heatmap_path"]):
                    if st.button(f"查看熱力圖 #{int(i)}", key=f"heatmap_{i}"):
                        st.session_state.selected_item = row
                        st.session_state.show_detail = True
                
                if st.button(f"詳細資訊 #{int(i)}", key=f"detail_{i}"):
                    st.session_state.selected_item = row
                    st.session_state.show_detail = True
                    # 設置一個標記，用於跳轉到詳細資訊區域
                    st.session_state.scroll_to_detail = True
    
    if "show_detail" in st.session_state and st.session_state.show_detail:
        show_detail_panel(st.session_state.selected_item)
        
        # 如果需要滾動到詳細資訊區域
        if st.session_state.get("scroll_to_detail", False):
            st.markdown("""
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    document.querySelector('#detail-section').scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                });
            </script>
            """, unsafe_allow_html=True)
            
            # 重置滾動標記，以便下次點擊
            st.session_state.scroll_to_detail = False

async def send_to_llm_api(payload):
    try:
        response = await fetch('http://localhost:8001/chat-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })

        if not response.ok:
            raise Exception(f"後端回應失敗：{response.status} {response.statusText}")

        data = await response.json()
        return data.response
    except Exception as e:
        st.error(f"呼叫後端聊天API時發生錯誤：{e}")
        return "抱歉，模型暫時無法回應。"

def show_detail_panel(item):
    # 創建一個錨點，以便自動滾動
    st.markdown("<div id='detail-section'></div>", unsafe_allow_html=True)
    with st.expander("詳細資訊", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            img = load_image(item["img_path"])
            if img:
                # 修改點 3: 將 use_column_width 替換為 use_container_width
                st.image(img, caption="原始圖片", use_container_width=True)
            
            # 修改點 4: 檢查是否為錯誤預測，且 heatmap_path 不為空
            correct = item["correct"] == 1
            
            # 顯示熱力圖相關的調試訊息
            st.write("預測狀態:", "錯誤" if not correct else "正確")
            if "heatmap_path" in item:
                heatmap_path = {item["heatmap_path"]}
                # st.write("熱力圖路徑:", item["heatmap_path"])
                
                if item["heatmap_path"] and os.path.exists(item["heatmap_path"]):
                    # st.write("熱力圖文件存在")
                    heatmap = load_image(item["heatmap_path"])
                    if heatmap:
                        # 修改點 3: 將 use_column_width 替換為 use_container_width
                        st.image(heatmap, caption="ScoreCAM 熱力圖", use_container_width=True)
                    else:
                        st.warning("熱力圖加載失敗")
                else:
                    st.warning("熱力圖路徑不存在或為空")
        
        with col2:
            st.subheader("分類結果")
            st.write(f"真實標籤: {item['class_name_true']}")
            st.write(f"預測標籤: {item['class_name_pred']}")
            st.write(f"預測是否正確: {'✓ 正確' if item['correct'] == 1 else '✗ 錯誤'}")
            
            st.subheader("預測概率")
            for cls in ["spaghetti", "ramen", "udon"]:
                st.write(f"{cls}: {item['probability_'+cls]*100:.2f}%")
            
            st.subheader("問 LLM 看看")
            user_input = st.text_input("請解釋為何錯誤？", key="user_input")
            # user_input = st.text_input("輸入您的問題", key="user_input")
            if st.button("發送"):
                with st.spinner("LLM 正在思考中..."):
                    # 將Pandas Series轉換為可序列化的字典
                    item_dict = {}
                    for k, v in item.items():
                        if isinstance(v, pd.Series):
                            item_dict[k] = v.to_dict()
                        else:
                            item_dict[k] = v
                    
                    # 構建符合API期望的請求結構
                    api_payload = {
                        "image_data": item_dict,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    }
                    # print(item_dict)
                    try:
                        response = requests.post(
                            "http://localhost:8001/chat-image",
                            headers={"Content-Type": "application/json"},
                            json=api_payload
                        )
                        
                        if response.status_code == 200:
                            llm_response = response.json().get("response", "無回應")
                        else:
                            llm_response = f"API 錯誤: {response.status_code}"
                    except Exception as e:
                        llm_response = f"連接錯誤: {str(e)}"
                    
                    st.write("LLM 回應:")
                    st.info(llm_response)
        if st.button("關閉"):
            st.session_state.show_detail = False
            st.rerun()

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'show_detail' not in st.session_state:
        st.session_state.show_detail = False
    if 'scroll_to_detail' not in st.session_state:
        st.session_state.scroll_to_detail = False
    
    # 修改點 1: 使用 tab 取代 sidebar.radio 進行頁面切換
    tab1, tab2 = st.tabs(["專案詳細說明", "分類結果"])
    
    with tab1:
        project_details()
    with tab2:
        results_page()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import joblib
from fastai.vision.all import *
from PIL import Image
import pickle
import pathlib
import io

# --- 1. เครื่องมือแกะกล่องพิเศษสำหรับ Python 3.13+ ---
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'WindowsPath':
            return pathlib.PosixPath
        return super().find_class(module, name)

class CustomPickle:
    __name__ = "pickle"
    Unpickler = SafeUnpickler
    def load(self, f, **kwargs):
        return SafeUnpickler(f).load()

custom_pickle = CustomPickle()

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_models():
    # โหลดโมเดลราคาบ้าน
    rf_data = joblib.load('house_price_rf.pkl')
    m = rf_data['model']
    cols = rf_data['columns']
    
    # โหลดโมเดล FastAI
    cnn = load_learner('room_classifier_fastai.pkl', pickle_module=custom_pickle)
    
    return m, cols, cnn

# เรียกใช้งานโมเดล (ประกาศตัวแปรให้ชัดเจน)
rf_model, rf_columns, cnn_model = load_models()

# --- 3. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI อสังหาริมทรัพย์", layout="centered")
st.title(" ระบบ AI อสังหาริมทรัพย์อัจฉริยะ ")
st.markdown("ยินดีต้อนรับ! เลือกใช้งาน AI || ดูไฟล์เอกสารกดปุ่มซ้ายบน ")

# Sidebar
st.sidebar.header("📥 ข้อมูลอ้างอิง (Dataset)")
with open("thailand_house_prices_1000.csv", "rb") as file:
    st.sidebar.download_button("ดาวน์โหลดไฟล์ CSV (ราคาบ้าน)", file, "thailand_house_prices.csv", "text/csv")
    
# เช็คไฟล์ ZIP ก่อนทำปุ่ม (เพื่อกัน Error ถ้าไฟล์ใหญ่เกินแล้วเราลบออก)
try:
    with open("House_Room_Dataset.zip", "rb") as file:
        st.sidebar.download_button("ดาวน์โหลดไฟล์ ZIP (ภาพห้อง)", file, "House_Room_Dataset.zip", "application/zip")
except FileNotFoundError:
    st.sidebar.warning("ไฟล์ ZIP ใหญ่เกินไป กรุณาโหลดผ่านลิงก์สำรอง")

# --- 4. ส่วนของ Tabs (แก้จาก tap เป็น tab ทั้งหมด) ---
tab1, tab2, tab3, tab4 = st.tabs(["💰 ประเมินราคาบ้าน", "📸 ทายภาพห้องจากรูป", "📁 อธิบาย 'ประเมินราคาบ้าน'", "📁 อธิบาย 'ทายภาพห้อง'"])

with tab1:
    loc_dict = {"ใจกลางกรุงเทพฯ (CBD)": "Bangkok_CBD", "ชานเมืองกรุงเทพฯ": "Bangkok_Suburb", "นนทบุรี": "Nonthaburi", "เชียงใหม่": "Chiang_Mai"}
    shape_dict = {"สี่เหลี่ยมจัตุรัส": "Square", "สี่เหลี่ยมผืนผ้า": "Rectangle", "รูปทรงไม่แน่นอน": "Irregular", "ชายธง": "Flag"}
    orient_dict = {"ทิศเหนือ": "North", "ทิศตะวันออก": "East", "ทิศใต้": "South", "ทิศตะวันตก": "West"}

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("ขนาดพื้นที่ (ตารางเมตร)", min_value=10.0, value=100.0)
        bed = st.number_input("จำนวนห้องนอน", min_value=0, value=3)
        bath = st.number_input("จำนวนห้องน้ำ", min_value=0, value=2)
    with col2:
        loc_thai = st.selectbox("ทำเลที่ตั้ง", list(loc_dict.keys()))
        shape_thai = st.selectbox("รูปแปลงที่ดิน", list(shape_dict.keys()))
        orient_thai = st.selectbox("ทิศทางหน้าบ้าน", list(orient_dict.keys()))
        
    if st.button("ประเมินราคาเลย!", type="primary"):
        input_data = pd.DataFrame({
            'Area_sqm': [area], 'Bedrooms': [bed], 'Bathrooms': [bath],
            'Location': [loc_dict[loc_thai]], 'Land_Shape': [shape_dict[shape_thai]], 'Orientation': [orient_dict[orient_thai]]
        })
        encoded_data = pd.get_dummies(input_data)
        # ใช้ rf_columns ที่ดึงมาจากฟังก์ชัน load_models
        ready_data = encoded_data.reindex(columns=rf_columns, fill_value=0)
        price = rf_model.predict(ready_data)[0]
        st.success(f"🎯 AI ประเมินราคาบ้านหลังนี้อยู่ที่: **{price:,.0f} บาท**")

with tab2:
    st.header("📸 ทายภาพห้องจากรูป")
    uploaded_file = st.file_uploader("เลือกรูปภาพห้อง...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
        if st.button("ให้ AI วิเคราะห์รูปภาพ"):
            img_fastai = PILImage.create(uploaded_file)
            pred_class, pred_idx, outputs = cnn_model.predict(img_fastai)
            st.success(f"🎯 AI มั่นใจ {outputs[pred_idx].item()*100:.2f}% ว่าคือ: **{pred_class.upper()}**")

with tab3:
    st.header("อธิบายการประเมินราคาบ้าน")
    st.markdown("...(ใส่เนื้อหาทฤษฎีของคุณได้เลย)...")

with tab4:
    st.header("อธิบายการทายภาพห้อง")
    st.markdown("...(ใส่เนื้อหาทฤษฎีของคุณได้เลย)...")
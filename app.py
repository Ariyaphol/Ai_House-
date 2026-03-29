import streamlit as st
import pandas as pd
import joblib
from fastai.vision.all import *
from PIL import Image
import pathlib
import platform

# --- ส่วนแก้ปัญหา Path และการโหลดโมเดล ---
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# สำคัญมาก: สร้างฟังก์ชันหลอกสำหรับ DataLoaders 
# เพื่อให้ fastai โหลดโมเดลผ่านโดยไม่สนเรื่อง dls เก่า
def label_func(x): return x  

@st.cache_resource
def load_models():
    
    rf_data = joblib.load('house_price_rf.pkl')
    m = rf_data['model']
    cols = rf_data['columns']
    
   
    cnn = load_learner('room_classifier_fastai.pkl')
    
    return m, cols, cnn


try:
    rf_model, rf_columns, cnn_model = load_models()
    model_ready = True
except Exception as e:
    st.error(f"⚠️ ระบบกำลังเตรียมโมเดล หรือเกิดข้อผิดพลาด: {e}")
    model_ready = False
# --------------------------------------




st.set_page_config(page_title="AI อสังหาริมทรัพย์", layout="centered")
st.title(" ระบบ AI อสังหาริมทรัพย์อัจฉริยะ ")
st.markdown("ยินดีต้อนรับ! เลือกใช้งาน AI || ดูไฟล์เอกสารกดปุ่มซ้ายบน ")

st.sidebar.header("📥 ข้อมูลอ้างอิง (Dataset)")
with open("thailand_house_prices_1000.csv", "rb") as file:
    st.sidebar.download_button("ดาวน์โหลดไฟล์ CSV (ราคาบ้าน)", file, "thailand_house_prices.csv", "text/csv")
 
try:
    with open("House_Room_Dataset.zip", "rb") as file:
        st.sidebar.download_button("ดาวน์โหลดไฟล์ ZIP (ภาพห้อง)", file, "House_Room_Dataset.zip", "application/zip")
except FileNotFoundError:
    st.sidebar.warning("ไฟล์ ZIP ใหญ่เกินไป กรุณาโหลดผ่านลิงก์สำรอง")


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
        
        input_dict = {
            'Area_sqm': [area],
            'Bedrooms': [bed],
            'Bathrooms': [bath],
            'Location': [loc_dict[loc_thai]], 
            'Land_Shape': [shape_dict[shape_thai]], 
            'Orientation': [orient_dict[orient_thai]]
        }
        input_data = pd.DataFrame(input_dict)

        
        encoded_data = pd.get_dummies(input_data)

     
        ready_data = encoded_data.reindex(columns=rf_columns, fill_value=0)

        try:
            price = rf_model.predict(ready_data)[0]
            st.balloons() # แสดงเอฟเฟกต์ฉลองความสำเร็จ
            st.success(f"🎯 AI ประเมินราคาบ้านหลังนี้อยู่ที่: **{price:,.0f} บาท**")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

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
    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.markdown("นำข้อมูลตาราง (CSV) มาแปลงข้อความให้เป็นตัวเลขด้วยเทคนิค One-Hot Encoding เพื่อให้คอมพิวเตอร์คำนวณได้ " \
    " \n\n แบ่งข้อมูลเป็น 2 ส่วน: สำหรับให้ AI ฝึกสอน (Train) 80% และสำหรับใช้ทดสอบความแม่นยำ (Test) 20%")

    st.header("2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)")
    st.markdown("ใช้เทคนิค Ensemble Learning (Voting Regressor) คือการนำ AI ผู้เชี่ยวชาญ 3 ตัวมาช่วยกันโหวตหาค่าเฉลี่ยราคาที่แม่นยำที่สุด ได้แก่:" \
         "\n\n  1) Random Forest: สร้างต้นไม้ตัดสินใจหลายๆ ต้นแล้วหาค่าเฉลี่ย" \
        "\n\n 2) Gradient Boosting: สร้างต้นไม้ที่เรียนรู้และแก้ไขข้อผิดพลาดของต้นก่อนหน้า " \
        "\n\n 3) Ridge Regression: ใช้สมการเชิงเส้นตรงที่ปรับลดความซับซ้อนเพื่อลดความผิดพลาด")
    
    st.header("3. ขั้นตอนการพัฒนา (Development Steps)")
    st.markdown("1) นำเข้าและทำความสะอาดข้อมูล -> แยกตัวแปรต้น (คุณสมบัติบ้าน) และตัวแปรตาม (ราคา)" \
                "\n\n 2) นำโมเดลทั้ง 3 ตัวมารวมเป็นคณะกรรมการ (Voting)" \
                "\n\n 3) สั่งฝึกสอน (Train) และวัดผลสอบด้วยค่าความคลาดเคลื่อน MAE, RMSE และค่าความแม่นยำ R-squared")
    
    st.header("4. แหล่งอ้างอิง (References)")
    st.markdown(" thailand_house_prices_1000.csv: เป็นข้อมูลที่เตรียมจากการเจนเนอเรตขึ้นมาใหม่ โดยมี 6 คุณสมบัติหลัก ได้แก่ ขนาดพื้นที่, จำนวนห้องนอน, จำนวนห้องน้ำ, ทำเลที่ตั้ง, รูปแปลงที่ดิน และทิศทางหน้าบ้าน พร้อมราคาขายจริง (Price_THB) ")


with tab4:
    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.markdown("รวบรวมรูปภาพห้อง 6 หมวดหมู่ และย่อขนาดภาพทุกรูปให้เท่ากันที่ 224x224 พิกเซล * แบ่งข้อมูลเป็น Train 80% และ Validation 20% (เพื่อเช็คไม่ให้ AI จำข้อสอบ)")

    st.header("2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)")
    st.markdown("ใช้โครงข่ายประสาทเทียม CNN (Convolutional Neural Network) เลียนแบบการมองเห็นของมนุษย์เพื่อสกัดจุดเด่นภาพ โดยใช้เทคนิค Transfer Learning คือการนำโมเดลชื่อ ResNet-18 (AI ที่เคยเรียนรู้รูปมาแล้วนับล้าน) มาสอนต่อยอดให้รู้จักแค่หมวดหมู่ห้องของเรา ช่วยให้เทรนได้เร็วและแม่นยำขึ้น")

    st.header("3. ขั้นตอนการพัฒนา (Development Steps)")
    st.markdown("1) โหลดรูปภาพผ่านระบบของ FastAI พร้อมทำ Data Augmentation (เช่น พลิก/ซูมภาพ)" \
                " \n\n 2) สร้างโมเดลโดยใช้ฐานสมอง ResNet-18" \
                " \n\n 3) สั่งฝึกสอนโมเดลด้วยคำสั่ง fine_tune() เพื่อปรับความเข้าใจให้เข้ากับรูปห้องของเรา" \
                "\n\n 4) ตรวจสอบความแม่นยำ (Accuracy) และบันทึกไฟล์สมอง (Export) ไปใช้งาน")

    st.header("4. แหล่งอ้างอิง (References)")
    st.markdown("ชุดข้อมูลรูปภาพจาก: Kaggle")
    st.link_button("คลิกไปที่ dataset จาก Kaggle", "https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset")
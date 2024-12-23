import streamlit as st
import numpy as np
import cv2
import joblib

# Tải mô hình đã lưu
svc_model = joblib.load('svc_model.pkl')

# Kích thước ảnh mà mô hình yêu cầu
IMG_SIZE = 227

# Hàm xử lý ảnh và dự đoán
def prepare_image(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize ảnh
    img = img.flatten()  # Chuyển ảnh thành vector 1D
    img = np.array([img])  # Tạo array 2D (số lượng ảnh, số pixel)
    return img

# Giao diện người dùng
st.title("Predict-Xray-Images-With-SVC-Model")
st.write("Chọn một tấm ảnh X-quang để dự đoán bệnh.")

# Upload ảnh
uploaded_file = st.file_uploader("Chọn tấm ảnh X-quang", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh tải lên
    image = uploaded_file.read()
    st.image(image, caption="Ảnh X-quang", use_column_width=True)

    # Lưu ảnh tạm thời để xử lý
    with open("temp_image.jpeg", "wb") as f:
        f.write(image)

    # Chuẩn bị ảnh và thực hiện dự đoán
    img = prepare_image("temp_image.jpeg")
    prediction = svc_model.predict(img)

    # Chuyển dự đoán thành nhãn (NORMAL hoặc PNEUMONIA)
    label = 'PNEUMONIA' if prediction[0] == 'PNEUMONIA' else 'NORMAL'

    # Hiển thị kết quả dự đoán
    st.subheader(f"Dự đoán: {label}")
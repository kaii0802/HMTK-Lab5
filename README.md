# HMTK-Lab5
Predict Xray Images With SVC Model

--- export mô hình SVC ---
import joblib
joblib.dump(svc, 'svc_model.pkl')

--- Di chuyển svc_model.pkl vào thư mục model_directory

--- chạy lệnh để cài đặt thư viện ---
pip install -r requirements.txt

---Tiếp theo chạy lệnh----
python app.py

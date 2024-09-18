import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Đường dẫn đến mô hình và ảnh
model_path = os.path.join(r'C:\Users\ADMIN\Desktop\DATN_Huy\PCB-component-detection\defect', 'best.torchscript')  # Thay 'path_to_model' bằng đường dẫn thực tế
image_path = os.path.join('./', 'pcb_10f_cc_2.png')  # Thay 'path_to_image' và 'your_image.jpg'

# Tải mô hình YOLO
model = YOLO(model_path)

# Dự đoán trên ảnh
results = model(image_path, imgsz=640, conf=0.25)

# Lấy ảnh gốc và hiển thị nhãn cùng với box
def display_image_with_boxes(image_path, results):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh từ BGR sang RGB

    # Lặp qua từng kết quả dự đoán
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Lấy tọa độ của bounding box
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]  # Độ tin cậy
            class_id = int(box.cls[0])  # ID của lớp
            class_name = model.names[class_id]  # Lấy tên lớp từ model
            
            # Vẽ bounding box trên ảnh
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Hiển thị tên lớp và độ tin cậy
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Hiển thị ảnh bằng matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # Tắt trục
    plt.show()

# Hiển thị ảnh với bounding box và nhãn
display_image_with_boxes(image_path, results)

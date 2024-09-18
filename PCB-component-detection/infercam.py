import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model đã huấn luyện (file .pt)
model = YOLO("./weights/best.pt")

# Mở webcam
cap = cv2.VideoCapture(1)  # Dùng 0 nếu là webcam tích hợp, hoặc 1 nếu là webcam cắm ngoài

if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Danh sách 25 màu sắc cố định cho 25 lớp
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192),
    (255, 128, 0)
]

# Khởi tạo cửa sổ matplotlib
plt.ion()  # Bật chế độ interactive mode để hiển thị thời gian thực
fig, ax = plt.subplots()

while True:
    # Đọc một khung hình từ webcam
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận dữ liệu từ webcam")
        break

    # Áp dụng model YOLO lên khung hình
    results = model.predict(source=frame, save=False, show=False)

    # Dictionary để đếm số lượng mỗi class
    class_counts = {}

    # Lấy các thông tin dự đoán (bounding boxes và nhãn)
    for result in results:
        for box in result.boxes:
            # Lấy thông tin về bounding box và nhãn lớp
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Kiểm tra và khởi tạo nếu class chưa tồn tại trong dictionary
            if class_name not in class_counts:
                class_counts[class_name] = 0

            # Cập nhật số lượng class
            class_counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ của bounding box

            # Lấy màu từ danh sách dựa trên class_id
            color = colors[class_id % len(colors)]

            # Vẽ bounding box và tên lớp với màu tương ứng
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Hiển thị số lượng mỗi class lên khung hình
    y_offset = 30
    for class_name, count in class_counts.items():
        if count > 0:  # Chỉ hiển thị những class có số lượng lớn hơn 0
            cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 30

    # Cập nhật khung hình trong matplotlib
    ax.clear()  # Xóa nội dung cũ
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Chuyển đổi khung hình từ BGR sang RGB
    plt.pause(0.001)  # Đợi một khoảng thời gian ngắn để cập nhật khung hình mới

    # Kiểm tra nếu người dùng nhấn 'q' để thoát
    if plt.waitforbuttonpress(timeout=0.001) and plt.get_current_fig_manager().canvas.toolbar.mode == '':
        break

# Giải phóng tài nguyên
cap.release()
plt.close(fig)  # Đóng cửa sổ matplotlib

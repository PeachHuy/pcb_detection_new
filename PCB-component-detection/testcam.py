import cv2

for i in range(5):  # Thử với các chỉ số từ 0 đến 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
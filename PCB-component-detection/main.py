from fastapi import (
    FastAPI,
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Response,
    Form,
    Depends,
    Query,
    WebSocket
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import io
import csv
import pandas as pd
import os
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from collections import Counter
import easyocr
import threading
import time
import json

app = FastAPI()

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins. Change to specific origins as needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods. Adjust if needed.
    allow_headers=["*"],  # Allow all headers. Adjust if needed.
)

# Biến này sẽ kiểm soát trạng thái của stream
stream_active = True    

# Thư mục chứa file CSV
CSV_DIRECTORY = "../check_pcb"  # Thay đổi đường dẫn này thành thư mục chứa file CSV của bạn

# Initialize the reader
reader = easyocr.Reader(['en']) 

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")

# Load the YOLO model
model = YOLO("./weights/best.pt")

class_counts = {}
# Threading lock
lock = threading.Lock()

def generate_frames():
    global class_counts
    # Mở webcam (dùng 0 nếu là webcam tích hợp, hoặc 1 nếu là webcam cắm ngoài)
    cap = cv2.VideoCapture(0)

    # Kiểm tra xem webcam có mở được không
    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    # Danh sách màu sắc cố định cho các lớp đối tượng
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
        (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192),
        (255, 128, 0)
    ]

    # Loop để liên tục đọc và xử lý từng khung hình từ webcam
    while True:
        ret, frame = cap.read()

        # Nếu không thể nhận được dữ liệu từ webcam thì dừng
        if not ret:
            print("Không thể nhận dữ liệu từ webcam")
            break

        # Áp dụng mô hình YOLO lên khung hình
        results = model.predict(source=frame, save=False, show=False)

        # Dictionary để lưu số lượng mỗi class
        class_counts = {}

        # Lấy thông tin dự đoán từ kết quả của YOLO
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Lấy ID của lớp dự đoán
                class_name = model.names[class_id]  # Tên lớp dự đoán
                
                # Đếm số lượng của mỗi lớp
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1

                # Tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Chọn màu từ danh sách dựa trên ID của class
                color = colors[class_id % len(colors)]

                # Vẽ bounding box và tên class lên khung hình
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Hiển thị số lượng mỗi class lên khung hình
        y_offset = 30
        for class_name, count in class_counts.items():
            if count > 0:  # Chỉ hiển thị nếu có đối tượng được phát hiện
                cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 30

        # Mã hóa khung hình thành định dạng JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Trả về khung hình dưới dạng luồng video
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Giải phóng tài nguyên khi kết thúc
    cap.release()

def resize_image(image: np.ndarray, max_edge_length: int) -> np.ndarray:
    """
    Resize the image so that the longer edge is equal to max_edge_length,
    maintaining the aspect ratio.
    """
    h, w = image.shape[:2]
    if h > w:
        scale = max_edge_length / h
        new_size = (int(w * scale), max_edge_length)
    else:
        scale = max_edge_length / w
        new_size = (max_edge_length, int(h * scale))

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_image


def read_imagefile(file) -> np.ndarray:
    """
    Read uploaded image file and convert it to OpenCV format.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        np.ndarray: Image in OpenCV format (BGR).
    """
    image = Image.open(BytesIO(file))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        # Predict on the image
        results = model.predict(source=image, conf=0.3, iou=0.7)
        # Extract prediction data
        predictions = []
        for box in results[0].boxes:
            predictions.append(
                {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bounding_box": {
                        "xmin": int(box.xyxy[0][0]),
                        "ymin": int(box.xyxy[0][1]),
                        "xmax": int(box.xyxy[0][2]),
                        "ymax": int(box.xyxy[0][3]),
                    },
                }
            )
        # Reverse mapping for convenience
        int_to_class_name = {v: k for k, v in class_name_to_int.items()}

        # Count occurrences
        counter = Counter(results[0].boxes.cls.tolist())

        # Initialize the result dictionary with all class names set to 0
        result = {class_name: 0 for class_name in class_name_to_int}

        # Update counts based on the actual occurrences in the list
        for num, count in counter.items():
            if num in int_to_class_name:
                result[int_to_class_name[int(num)]] = count
    
        response = JSONResponse(content={
            "image_shape": {
                "height": results[0].orig_shape[0],
                "width": results[0].orig_shape[1],
            },
            "speed": results[0].speed,
            "appearances": result,
            "predictions": predictions,
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the class name to integer mapping
class_name_to_int = {
    "R": 0,
    "C": 1,
    "U": 2,
    "Q": 3,
    "J": 4,
    "L": 5,
    "RA": 6,
    "D": 7,
    "RN": 8,
    "TP": 9,
    "IC": 10,
    "P": 11,
    "CR": 12,
    "M": 13,
    "BTN": 14,
    "FB": 15,
    "CRA": 16,
    "SW": 17,
    "T": 18,
    "F": 19,
    "V": 20,
    "LED": 21,
    "S": 22,
    "QA": 23,
    "JP": 24
}

def map_classes_to_int(classes_list):
    return [class_name_to_int[cls] for cls in classes_list if cls in class_name_to_int]

@router.post("/predict-png")
async def predict_png(
    file: UploadFile = File(...),
    show_conf: Optional[bool] = Form(True),
    show_labels: Optional[bool] = Form(True),
    show_boxes: Optional[bool] = Form(True),
    line_width: Optional[int] = Form(None),
    classes: Optional[str] = Form([]),
):
    try:
        if line_width is None:
            line_width = 10

        # Read the uploaded file
        image = read_imagefile(await file.read())

        # Predict on the image
        if classes:
            results = model.predict(source=image, conf=0.3, iou=0.7, classes=map_classes_to_int(classes))
        else:
            results = model.predict(source=image, conf=0.3, iou=0.7)
        
        # Extract prediction data
        img_with_boxes = results[0].plot(
            boxes=show_boxes,
            labels=show_labels,
            conf=show_conf,
            line_width=line_width,
        )

        # Encode the image to PNG format
        _, img_encoded = cv2.imencode(".png", img_with_boxes)
        img_bytes = img_encoded.tobytes()

        # Create the response with the encoded image
        response = Response(content=img_bytes, media_type="image/png")
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-ocr")
async def predict_ocr(
    file: UploadFile = File(...),
    show_ocr: Optional[bool] = Form(True),
):
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        if show_ocr:
            ocr_results = reader.readtext(image)
            for (bbox, text, prob) in ocr_results:
                top_left = tuple(map(int, bbox[0]))  # Convert [x1, y1] to integers
                bottom_right = tuple(map(int, bbox[2]))  # Convert [x3, y3] to integers

                # Draw the bounding box
                image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
                
                # Put the recognized text above the bounding box
                image = cv2.putText(
                    image,
                    text,
                    (top_left[0], top_left[1] - 10),  # Position the text slightly above the box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,  # Font scale
                    (0, 255, 0),  # Text color (green)
                    2,  # Thickness
                    cv2.LINE_AA,
                )

        _, img_encoded = cv2.imencode(".png", image)
        img_bytes = img_encoded.tobytes()
        response = Response(content=img_bytes, media_type="image/png")
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lấy danh sách file csv
@router.get("/csv-files")
async def get_csv_files():
    try:
        # Lấy danh sách các file trong thư mục
        files = os.listdir(CSV_DIRECTORY)

        # Lọc file CSV
        csv_files = [file for file in files if file.endswith(".csv")]

        # Trả về danh sách file CSV dưới dạng JSON
        response = JSONResponse(content={"csv_files": csv_files})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Đọc csv
# @app.get("/csv")
@router.get("/csv")
async def read_csv(file_name: str = Query(..., description="Tên file CSV")):
    try:
        # Đường dẫn tới file CSV
        base_path = "C:/Users/ADMIN/Desktop/DATN_Huy/check_pcb/"
        file_path = os.path.join(base_path, file_name)
        
        # Kiểm tra nếu file không tồn tại
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="CSV file not found.")
        
        # Đọc file CSV, sử dụng ký tự phân tách là '\t'
        df = pd.read_csv(file_path, header=None, sep='\t')
        print(df)  # In dữ liệu CSV để kiểm tra
        
        # Kiểm tra định dạng file CSV
        if df.shape[1] < 3:  # Nếu file không có ít nhất 3 cột
            raise HTTPException(status_code=400, detail="CSV file does not have enough columns.")
        
        # Xử lý và chuyển đổi dữ liệu theo định dạng yêu cầu
        data = []
        for index, row in df.iterrows():
            symbol = row[0]  # Ký hiệu từ cột đầu tiên
            description = row[1]  # Mô tả từ cột thứ hai
            value = row[2]  # Giá trị từ cột thứ ba
            
            # Thêm vào danh sách dữ liệu
            data.append({
                "symbol": symbol,
                "description": description,
                "value": value
            })
        
        # Trả về dữ liệu và thêm headers CORS
        response = JSONResponse(content=data)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


# Endpoint trả về luồng video nhận diện
@router.get("/video_feed")
async def video_feed():
    # return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
    global stream_active
    if stream_active:
        return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(content="Stream has been stopped", media_type="text/plain")
    
# API để dừng stream
@router.post("/stop_stream")
async def stop_stream():
    global stream_active
    stream_active = False  # Dừng stream
    return {"message": "Stream stopped"}

#Số lượng mỗi nhãn khi stream
@app.get("/api/v1/class_counts")
async def get_class_counts():
    return class_counts


app.include_router(router)

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import threading

app = FastAPI()

# Mở webcam
cap = cv2.VideoCapture(0)

# Threading lock
lock = threading.Lock()

def generate_frames():
    while True:
        with lock:
            ret, frame = cap.read()
            if not ret:
                continue

            # Encode frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        # Trả về khung hình dưới dạng luồng video
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

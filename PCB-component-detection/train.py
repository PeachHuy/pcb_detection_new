from ultralytics import YOLO, settings

settings.update({"wandb": False})

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8m.pt")  # yolov8n.pt, yolov8m.pt, etc.

# Train the model
results = model.train(
    data="./data.yaml",  # Direct dataset details
    epochs=50,  # Number of epochs to train for
    imgsz=1280,  # Image size for training
    batch=4,  # Batch size for training
)

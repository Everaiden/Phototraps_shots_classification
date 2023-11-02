from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict('./Datasets/imgs', save=True)
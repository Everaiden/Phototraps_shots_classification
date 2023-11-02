from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='.\Datasets\data.yaml', epochs=2)
results = model.val()
results = model.predict('./Datasets/imgs', save=True)
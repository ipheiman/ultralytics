# YOLO(yolov8s-p2.pt")

from ultralytics import YOLO

# Load a model
'''YOLOv8n'''
model = YOLO('yolov8-p2.yaml').load('yolov8s.pt')
'''YOLOv8s'''
# model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# If from scratch, load yaml file

# Finetune
''' imgsz = 640'''
# model.train(data="fpic.yaml", epochs=1000, batch=32, device=[1,2])  # train the model
''' imgsz = 1280'''
model.train(data="fpic.yaml", epochs=1000, batch=4, imgsz=1280, device=[1,2])
''' imgsz = 2560'''
# model.train(data="fpic.yaml", epochs=1000, batch=2, imgsz=2560, device=[1,2])

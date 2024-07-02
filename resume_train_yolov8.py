from ultralytics import YOLO

model = YOLO('runs/detect/exp5_TRAIN_yolov8s_ufl_1280/weights/last.pt').train(resume=True)
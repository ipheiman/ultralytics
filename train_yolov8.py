from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# If from scratch, load yaml file

# Finetune
model.train(data="fpic.yaml", epochs=1000, batch=64, device=[0,1,2,3])  # train the model

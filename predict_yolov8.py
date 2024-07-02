from ultralytics import YOLO
import argparse
import os
import cv2
from tqdm import tqdm

idx2class = {
0: "R",
1: "C",
2: "L",
3: "IC",
4: "Others"
}

def draw_boxes(image, boxes, confs, clss, class_names):
    for i in range(len(boxes)):
        box = boxes[i].int().tolist()  # Convert to list of integers
        confidence = float(confs[i])
        class_id = int(clss[i])

        x1, y1, x2, y2 = box
        label = f"{class_names[class_id]}: {confidence:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put text
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main(model_path, source_dir, dest_dir):
    # Load a model
    model = YOLO(model_path)  # pretrained YOLOv8n model

    images = [file for file in os.listdir(source_dir) if file.endswith(".png")]
    images = [os.path.join(source_dir, file) for file in images]
    results = model(images) 

    with tqdm(total=len(results)) as pbar:
        for idx, r in enumerate(results):

            # For each image, extract detection results
            boxes = r.boxes.xyxy
            confidences = r.boxes.conf
            classes = r.boxes.cls
            image = cv2.imread(images[idx])

            image_with_boxes = draw_boxes(image, boxes, confidences, classes, idx2class)
            
            # Save the annotated image
            output_path = os.path.join(dest_dir, os.path.basename(images[idx]))
            cv2.imwrite(output_path, image_with_boxes)
            pbar.update(1)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Predict with YOLOv8 model')
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="The path of YOLOv8 model")
    
    parser.add_argument('-i',
                        "--data_dir",
                        type=str,
                        required=True,
                        help="The path of directory containing images to predict on")    

    parser.add_argument('-id',
                        '--images_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where predicted images will be saved")
    
    args = parser.parse_args()

    # create directories if not exist
    if not os.path.exists(args.images_dest_dir):
        os.makedirs(args.images_dest_dir)
    main(args.model, args.data_dir, args.images_dest_dir)
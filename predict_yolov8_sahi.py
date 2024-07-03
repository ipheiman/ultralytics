from ultralytics import YOLO
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi import AutoDetectionModel
import os
import argparse
import cv2
from tqdm import tqdm
from create_yolo_annotations import convert_to_yolo


def draw_boxes(image, predictions):
    for i in range(len(predictions)):
        bbox = predictions[i].bbox  # Convert to list of integers
        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)
     
        category_name = predictions[i].category.name
        confidence = float(predictions[i].score.value)
        label = f"{category_name}: {confidence:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put text
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main(model_dir, source_img_dir, dest_annotation, dest_img, draw_box = False):

    if not os.path.exists(dest_annotation):
        os.makedirs(dest_annotation)

    if draw_box and not os.path.exists(dest_img):
        os.makedirs(dest_img)
        
    # Initialize the model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_dir,
        confidence_threshold=0.4,
        device="cuda:1" 
    )

    source_files = os.listdir(source_img_dir)
    with tqdm(total=len(source_files)//2) as pbar:
        for file in source_files:
            if file.endswith(".png"):
                image= os.path.join(source_img_dir, file)
                sliced_results = get_sliced_prediction(
                    image = image,
                    detection_model = detection_model,
                    slice_height=256,
                    slice_width=256,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )
                predictions = sliced_results.object_prediction_list
                img_array = cv2.imread(image)
                img_h, img_w, _ = img_array.shape

                # Write predictions
                annotation_file = open(os.path.join(dest_annotation, os.path.basename(file)+".txt"),"w")
                for i in range(len(predictions)):
                    bbox = predictions[i].bbox  # Convert to list of integers
                    x1 = int(bbox.minx)
                    y1 = int(bbox.miny)
                    x2 = int(bbox.maxx)
                    y2 = int(bbox.maxy)
                
                    category_id = predictions[i].category.id           
                    (x_center, y_center, w, h) = convert_to_yolo(x1,y1,x2-x1,y2-y1, img_w, img_h)
                    annotation_file.write(f"{category_id} {x_center} {y_center} {w} {h}\n")
                annotation_file.close()


                if draw_box:
                    predicted_image = draw_boxes(img_array, predictions)
                    # Save the annotated image
                    output_path = os.path.join(dest_img, file)
                    cv2.imwrite(output_path, predicted_image)
            pbar.update(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Predict with YOLOv8 model with SAHI')
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="The path of YOLOv8 model")

    parser.add_argument("--source_img",
                        type=str,
                        required=True,
                        help="The path of images to run inference on")    

    parser.add_argument("--dest_annotation",
                        type=str,
                        help="The path to save predicted annotations")    

    parser.add_argument("--dest_img",
                        type=str,
                        help="The path to save predicted images")    
    
    parser.add_argument("--draw_box",
                        action='store_true',
                        help="Save predicted images")    
    
    args = parser.parse_args()
    
    main(args.model, args.source_img, args.dest_annotation, args.dest_img, args.draw_box)
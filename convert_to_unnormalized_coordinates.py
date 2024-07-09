import os
import cv2

# gt: /home/heiman/L3/DATASET/UFL_PCB_FULL/yolo_training_pascal_format/test/
# det: /home/heiman/L3/CODE/ultralytics/runs/detect/exp1_fpic_test_SAHI/
def convert_to_unnormalized(x_center, y_center, w, h, image_width, image_height):
    # Calculate bounding box coordinates in pixels
    x = int((x_center - w / 2.0) * image_width)
    y = int((y_center - h / 2.0) * image_height)
    w = int(w * image_width)
    h = int(h * image_height)
    
    return x, y, w, h

def main():
    gt_dir = "/home/heiman/L3/DATASET/UFL_PCB_FULL/yolo_training/test/"
    gt_dir_pascalformat = "/home/heiman/L3/DATASET/UFL_PCB_FULL/yolo_training_pascal_format/test/"

    if not os.path.exists(gt_dir_pascalformat):
        os.makedirs(gt_dir_pascalformat)

    files = os.listdir(gt_dir)
    for file in files:
        if file.endswith(".txt"):
            name, _ = os.path.splitext(file)
            annotation_file = open(os.path.join(gt_dir_pascalformat, name+".txt"),"w")

            img = cv2.imread(os.path.join(gt_dir, name+".png"))
            img_h, img_w, _ = img.shape

            with open(os.path.join(gt_dir,file), "r") as f:
                lines = f.readlines()

                # list of lists
                lines = [line.strip().split() for line in lines]

                for line in lines:
                    # line is a list
                    class_id = int(line[0])
                    x_center = float(line[1])
                    y_center = float(line[2])
                    w = float(line[3])
                    h = float(line[4])    
                    x, y, w, h = convert_to_unnormalized(x_center, y_center, w, h, img_w, img_h)
                    annotation_file.write(f"{class_id} {x} {y} {x+w} {y+h}\n")
                annotation_file.close()

                    




    


if __name__ == "__main__":
        main()
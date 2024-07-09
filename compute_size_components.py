import argparse
import os
import cv2
from tqdm import tqdm
from create_yolo_annotations import convert_to_unnormalized
import pandas as pd

    # Read in annotation file (YOLO format)
    # Read in image

def main(images_dir, annotation_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Fetch annotation, read corresponding image, put bounding box
    annotation_files = os.listdir(annotation_dir)
    annotation_files = [file for file in annotation_files if file.endswith(".txt")]

    data = []

    with tqdm(total=len(annotation_files)) as pbar:
        for file in annotation_files:
            yolo_annotation_filename = os.path.join(annotation_dir, file)
            name, _ = os.path.splitext(file)
            if os.path.exists(os.path.join(images_dir, name+".png")):
                img = cv2.imread(os.path.join(images_dir, name+".png"))
                img_h, img_w, img_c = img.shape

                with open(yolo_annotation_filename, "r") as f:
                    lines = f.readlines()
                    # list of lists
                    lines = [line.strip().split() for line in lines]

                    for line in lines:
                        # line is a list
                        # YOLO format
                        class_id = int(line[0])
                        x_center = float(line[1])
                        y_center = float(line[2])
                        w = float(line[3])
                        h = float(line[4])    
                        _, _, ori_w, ori_h = convert_to_unnormalized(x_center, y_center, w, h, img_w, img_h)
                        data.append([class_id, ori_h, ori_w])
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Component', 'Height', 'Width'])

    # Save DataFrame
    df.to_csv(os.path.join(dest_dir, "size_of_components.csv"), index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Compute size (height and width in pixels) of components")
    parser.add_argument('-i',
                        "--images_dir",
                        type=str,
                        required=True,
                        help="The path of train input images")
    parser.add_argument('-a',
                        "--annotation_dir",
                        type=str,
                        required=True,
                        help="The path of train annotations")    
    parser.add_argument('-d',
                    '--dest_dir',
                    type=str,
                    required=True,
                    help="The path of directory to write size of components")
    
    args = parser.parse_args()

    main(args.images_dir, args.annotation_dir, args.dest_dir)



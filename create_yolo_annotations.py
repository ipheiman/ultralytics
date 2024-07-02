# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# --------------------------------------------------------------------

# Adapted from CandleLabAI code

"""
Create YOLO annotations for FPIC dataset. Save overlaid YOLO bounding boxes on dest_image folder.

python create_yolo_annotations.py -i ../../DATASET/UFL_PCB_SAMPLE/pcb_image/ -a ../../DATASET/UFL_PCB_SAMPLE/smd_annotation/ -ad ../../DATASET/UFL_PCB_SAMPLE/yolo_annotations -id ../../DATASET/UFL_PCB_SAMPLE/plotted_yolo_pcb_image

"""

from glob import glob
import argparse
import sys
import ast
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import random
import shutil 

# color_values is used to encode mask into one hot mapping.
color_values = {
    "R": (255, 0, 0), # 0: Resistor
    "C": (255, 255, 0), # 1: Capacitor
    "U": (0, 234, 255), # 2: Integrated Circuit
    "Q": (170, 0, 255),
    "J": (255, 127, 0),
    "L": (191, 255, 0), # 5: Inductor
    "RA": (0, 149, 255),
    "D": (106, 255, 0),
    "RN": (0, 64, 255),
    "TP": (237, 185, 185),
    "IC": (185, 215, 237), # 10: Integrated Circuit
    "P": (231, 233, 185),
    "CR": (220, 185, 237),
    "M": (185, 237, 224),
    "BTN": (143, 35, 35),
    "FB": (35, 98, 143),
    "CRA": (143, 106, 35),
    "SW": (107, 35, 143),
    "T": (79, 143, 35),
    "F": (115, 115, 115),
    "V": (204, 204, 204),
    "LED": (245, 130, 48),
    "S": (220, 190, 255),
    "QA": (170, 255, 195),
    "JP": (255, 250, 200)
}

# Custom for passive components
class2idx = {
    "R": 0, # resistor
    "C": 1, # capacitor
    "L": 2, # inductor
    "IC": 3, # chip
    "U": 3, # chip
    "Others": 4
}

# for idx, class_name in enumerate(color_values):
#     class2idx[class_name] = idx


# Util functions
def convert_to_yolo(x, y, w, h, image_width, image_height):
    """
    Convert bounding box coordinates to YOLO format.
    
    Parameters:
    - x, y: Top-left corner coordinates of the bounding box.
    - w, h: Width and height of the bounding box.
    - image_width, image_height: Dimensions of the image.
    
    Returns:
    - yolo_box: Tuple containing (x_center, y_center, normalized_width, normalized_height)
                in YOLO format.
    """
    # Calculate center of the bounding box
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    
    # Normalize coordinates and dimensions
    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height
    
    return (x_center, y_center, w, h)          

def convert_to_unnormalized(x_center, y_center, w, h, image_width, image_height):
    # Calculate bounding box coordinates in pixels
    x = int((x_center - w / 2.0) * image_width)
    y = int((y_center - h / 2.0) * image_height)
    w = int(w * image_width)
    h = int(h * image_height)
    
    return x, y, w, h


# Main fuctions
def prepare_data(source_image_dir,
                 source_annotation_dir,
                 dest_annotation_dir
):
    """
    Helper function which creates masks and croops
    Args:
        source_image_dir: image directory containing input images
        source_annotation_dir: annotation directory containing csv annotations
        dest_annotation_dir: destination directory for storing yolo annotations

    """
    annotations_list = glob(os.path.join(source_annotation_dir, "*.csv"))
    with tqdm(total=len(annotations_list)) as pbar:
        for annotation in annotations_list:
            df = pd.read_csv(annotation)
            # checking if at least 1 designation is present in annotation
            if df["Designator"].isna().sum() != df.shape[0]:
                image_name = list(df["Image File"].unique())
                if os.path.exists(os.path.join(source_image_dir, image_name[0])):

                    # Create annotation txt file
                    name, _ = os.path.splitext(image_name[0])
                    annotation_file = open(os.path.join(dest_annotation_dir,name+".txt"),"w")
                    img = cv2.imread(os.path.join(source_image_dir, image_name[0]))
                    img_h, img_w, img_c = img.shape
                    # mask = np.zeros(shape=img.shape, dtype=np.uint8)
                    vertices_list = list(df["Vertices"])
                    designator_list = list(df["Designator"])
                    for (anote, cat) in zip(vertices_list, designator_list):
                        if cat in color_values:
                            color_code = color_values[cat]

                            if cat not in class2idx:
                                class_idx = 4 # Others
                            else:
                                class_idx = class2idx[cat]

                        else:
                            continue
                        try:
                            pts = np.array(ast.literal_eval(anote))[0].reshape((-1, 1, 2))
                        except:
                            continue

                        # create mask
                        mask = np.zeros(shape=img.shape, dtype=np.uint8)
                        mask = cv2.polylines(mask, [pts], True, color_code, 2)
                        mask = cv2.fillPoly(mask, [pts], color=color_code)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)
                        x,y,w,h = cv2.boundingRect(contours[0])
                        (x_center, y_center, w, h) = convert_to_yolo(x,y,w,h, img_w, img_h)
                        # print(x,y,w,h, img_w, img_h)

                        annotation_file.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")
                annotation_file.close()
                
            pbar.update(1)


def check_yolo_annotations(source_image_dir, dest_annotation_dir, images_dest_dir):
    '''
    Input: source image, yolo annotations
    Output: bounding boxed images
    '''
    idx2class = {
    0: "R",
    1: "C",
    2: "L",
    3: "IC",
    4: "Others"
}

    # Fetch annotation, read corresponding image, put bounding box
    annotation_files = os.listdir(dest_annotation_dir)
    annotation_files = [file for file in annotation_files if file.endswith(".txt")]
    print(annotation_files)

    with tqdm(total=len(annotation_files)) as pbar:
        for file in annotation_files:
            yolo_annotation_filename = os.path.join(dest_annotation_dir, file)
            name, _ = os.path.splitext(file)
            if os.path.exists(os.path.join(source_image_dir, name+".png")):
                img = cv2.imread(os.path.join(source_image_dir, name+".png"))
                img_h, img_w, img_c = img.shape

                with open(yolo_annotation_filename, "r") as f:
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
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) 

                        class_label = idx2class[class_id]
                        cv2.putText(img, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 165, 0), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(images_dest_dir,name+".png"),img)
                        
            pbar.update(1)


def create_train_val_test(source_image_dir, dest_annotation_dir, split_dir):
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # Create train val test on split dir
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)    

    # List all annotation files in the dest_annotation_dir
    annotation_files = os.listdir(dest_annotation_dir)
    
    # Shuffle the list of annotation files
    random.shuffle(annotation_files)

    # Calculate the number of files for each split
    total_files = len(annotation_files)
    num_train = int(total_files * train_ratio)
    num_val = int(total_files * val_ratio)
    num_test = total_files - num_train - num_val

    # Split the files
    train_files = annotation_files[:num_train]
    val_files = annotation_files[num_train:num_train + num_val]
    test_files = annotation_files[num_train + num_val:]
      
    # Copy files to their respective directories
    for file_list, split_dir in zip([train_files, val_files, test_files], [train_dir, val_dir, test_dir]):
        with tqdm(total=len(file_list)) as pbar:
            for file_name in file_list:
                # Copy annotation file
                source_file = os.path.join(dest_annotation_dir, file_name)
                dest_file = os.path.join(split_dir, file_name)
                shutil.copy(source_file, dest_file)
                
                # Copy image file
                image_file = os.path.join(source_image_dir, os.path.splitext(file_name)[0] + '.png')  # Change extension as needed
                if os.path.exists(image_file):
                    dest_file = os.path.join(split_dir, os.path.splitext(file_name)[0] + '.png')
                    shutil.copy(image_file, dest_file)    
                pbar.update(1)

def main(source_image_dir,
         source_annotation_dir,
         dest_annotation_dir,
         images_dest_dir,
         split_dir
         ):
    """
    main function which creates mask
    Args:
        source_image_dir: image directory containing input images
        source_annotation_dir: annotation directory containing csv annotations
        dest_annotation_dir: destination directory for storing yolo annotations
    """
    # create directories if not exist
    if not os.path.exists(dest_annotation_dir):
        os.makedirs(dest_annotation_dir)
    prepare_data(source_image_dir,
            source_annotation_dir,
            dest_annotation_dir
            )

    if not os.path.exists(images_dest_dir):
        os.makedirs(images_dest_dir)
    check_yolo_annotations(source_image_dir, dest_annotation_dir, images_dest_dir)

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    create_train_val_test(source_image_dir, dest_annotation_dir, split_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Create YOLO annotations')
    parser.add_argument('-i',
                        "--images_dir",
                        type=str,
                        required=True,
                        help="The path of directory containing input images")
    parser.add_argument('-a',
                        '--annotations_dir',
                        type=str,
                        required=True,
                        help="The path of directory containing annotations")
    
    parser.add_argument('-ad',
                        '--yolo_annotations_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where yolo annotations are stored")
    parser.add_argument('-id',
                        '--images_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where bounding boxed images are stored")
    
    parser.add_argument('--split',
                        type=str,
                        required=True,
                        help="The path of directory containing train val test split for YOLO training")
    
    args = parser.parse_args()


    main(source_image_dir = args.images_dir,
         source_annotation_dir = args.annotations_dir,
         dest_annotation_dir = args.yolo_annotations_dest_dir,
         images_dest_dir = args.images_dest_dir,
         split_dir = args.split)

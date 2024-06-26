# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# --------------------------------------------------------------------

# Adapted from CandleLabAI code

"""
Create YOLO annotations for FPIC dataset

Usage: python create_mask.py -i ../../data/pcb_image/ -a ../../data/smd_annotation/ -id ../../data/segmentation/images -ad ../../data/segmentation/masks -cd ../../data/classification/images/
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

# color_values is used to encode mask into one hot mapping.
color_values = {
    "R": (255, 0, 0),
    "C": (255, 255, 0),
    "U": (0, 234, 255), 
    "Q": (170, 0, 255),
    "J": (255, 127, 0),
    "L": (191, 255, 0),
    "RA": (0, 149, 255),
    "D": (106, 255, 0),
    "RN": (0, 64, 255),
    "TP": (237, 185, 185),
    "IC": (185, 215, 237),
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

class2idx = {}
for idx, class_name in enumerate(color_values):
    class2idx[class_name] = idx

idx2class = {idx:class_name for class_name, idx in class2idx.items()}

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

                # Create annotation txt file
                annotation_filename = os.path.basename(annotation)
                name, ext = os.path.splitext(annotation_filename)
                annotation_file = open(os.path.join(dest_annotation_dir,name+".txt"),"w")
                image_name = list(df["Source Image Filename"].unique())
                if os.path.exists(os.path.join(source_image_dir, image_name[0])):
                    img = cv2.imread(os.path.join(source_image_dir, image_name[0]))
                    img_h, img_w, img_c = img.shape
                    # mask = np.zeros(shape=img.shape, dtype=np.uint8)
                    vertices_list = list(df["Vertices"])
                    designator_list = list(df["Designator"])
                    for (anote, cat) in zip(vertices_list, designator_list):
                        if cat in color_values:
                            color_code = color_values[cat]
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

def main(source_image_dir,
         source_annotation_dir,
         dest_annotation_dir
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

    # if not os.path.exists(dest_masks_sir):
    #     os.makedirs(dest_masks_sir)

    # if not os.path.exists(dest_crops_dir):
    #     os.makedirs(dest_crops_dir)

    prepare_data(source_image_dir,
                 source_annotation_dir,
                 dest_annotation_dir
                 )

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
    # parser.add_argument('-id',
    #                     '--images_dest_dir',
    #                     type=str,
    #                     required=True,
    #                     help="The path of destination directory where images needs to be stored")
    parser.add_argument('-ad',
                        '--yolo_annotations_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where yolo annotations are stored")
    # parser.add_argument('-cd',
    #                     '--crops_dest_dir',
    #                     type=str,
    #                     required=True,
    #                     help="The path of destination directory where crops needs to be stored")
    args = parser.parse_args()


    main(source_image_dir = args.images_dir,
         source_annotation_dir = args.annotations_dir,
         dest_annotation_dir = args.yolo_annotations_dest_dir)

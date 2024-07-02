"""
This program can be used to extract valuable information from the pcb wacv 2019 dataset: https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection
"""

import sys
import os
import xml.etree.ElementTree as ET
import cv2
import argparse
from tqdm import tqdm


datasetPath = "pcb_wacv_2019/"
destinationPath = "pcb_wacv_2019_annotated"

if not os.path.exists(destinationPath):
    os.makedirs(destinationPath)
xmlPath = None

formattedDatasetPath = "pcb_wacv_2019_formatted/"

# Custom for passive components
class2idx = {
    "resistor": 0, # resistor
    "capacitor": 1, # capacitor
    "inductor": 2, # inductor
    "ic": 3, # chip
}

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


def main():
    
    # 47 boards
    with tqdm(total=len(os.listdir(datasetPath))) as pbar:
        for folder in os.listdir(datasetPath):
            folderPath = os.path.join(datasetPath, folder)
            if os.path.isdir(folderPath):
                for file in os.listdir(folderPath):
                    if file.endswith(".xml"):
                        xmlPath = os.path.join(folderPath, file)

            
            if not xmlPath:
                print("[WW] xml not recogonized, skip!")
                continue
            
            # load xml
            # print(f"Proccesing xml: {xmlPath}")
            tree = ET.parse(xmlPath)
            root = tree.getroot()

            image_width = int(root[4][0].text)
            image_height = int(root[4][1].text)


            # Create annotation file
            xml_dir = os.path.dirname(xmlPath)
            filename = os.path.basename(xmlPath)
            filename, _ = os.path.splitext(filename)
            annotation_file = open(os.path.join(xml_dir, filename+".txt"),"w")

            # In each XML file
            for i in range(6,len(list(root))):
                compName = root[i][0].text

                # Find out which component in compName
                for idx, name in enumerate(class2idx.keys()):
                    if name in compName:
                        class_label = idx
                        break

                    else:
                        class_label = 4 #others

                if "text" in compName:
                    continue
                bndbox = root[i][4]
                Xmin = int(bndbox[0].text)
                Ymin = int(bndbox[1].text)
                Xmax = int(bndbox[2].text)
                Ymax = int(bndbox[3].text)


                x_center, y_center, w, h = convert_to_yolo(Xmin, Ymin, Xmax-Xmin, Ymax-Ymin, image_width, image_height)
                annotation_file.write(f"{class_label} {x_center} {y_center} {w} {h}\n")
            annotation_file.close()
            # print(f"Written to: {filename}")
            pbar.update(1)

# '''Check annotations'''

    idx2class = {
    0: "R",
    1: "C",
    2: "L",
    3: "IC",
    4: "Others"
}

    with tqdm(total=len(os.listdir(datasetPath))) as pbar:

        for folder in os.listdir(datasetPath):
            folderPath = os.path.join(datasetPath, folder)
            if os.path.isdir(folderPath):
                imagefilename = folder+".jpg"

                if not os.path.exists(os.path.join(folderPath, imagefilename)):
                    print(f"{imagefilename} DOES NOT EXIST!")
                    continue # Dont need to draw annotations on the image file anymore.
                img = cv2.imread(os.path.join(folderPath,imagefilename))
                img_h, img_w, _ = img.shape

                annotation_file = folder+".txt"

                with open(os.path.join(folderPath, annotation_file), "r") as f:
                    lines = f.readlines()
                    # list of lists
                    lines = [line.strip().split() for line in lines]

                    for line in lines:
                        class_id = int(line[0])
                        x_center = float(line[1])
                        y_center = float(line[2])
                        w = float(line[3])
                        h = float(line[4])    
                        x, y, w, h = convert_to_unnormalized(x_center, y_center, w, h, img_w, img_h)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1) 

                        class_label = idx2class[class_id]
                        cv2.putText(img, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    name, _ = os.path.splitext(imagefilename)
                    cv2.imwrite(os.path.join(destinationPath, name+"_annotated"+".jpg"),img)
            pbar.update(1)  


def convert_to_unnormalized(x_center, y_center, w, h, image_width, image_height):
    # Calculate bounding box coordinates in pixels
    x = int((x_center - w / 2.0) * image_width)
    y = int((y_center - h / 2.0) * image_height)
    w = int(w * image_width)
    h = int(h * image_height)
    
    return x, y, w, h


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

            



if __name__ == "__main__":
    main()
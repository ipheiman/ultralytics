import os
import argparse
from tqdm import tqdm
import cv2
from create_yolo_annotations import check_yolo_annotations


def convert_classes(source_labels_dir, dest_labels_dir):
    dirs = os.listdir(source_labels_dir)

    for d in dirs:
        # If d is a directory
        if os.path.isdir(os.path.join(source_labels_dir,d)):
            dir_path = os.path.join(source_labels_dir, d)
            annotation_files = os.listdir(dir_path)

            with tqdm(total=len(annotation_files)) as pbar:
                for file in annotation_files:
                    filepath = os.path.join(source_labels_dir,d,file)

                    with open(filepath, "r") as f:
                        lines = f.readlines()
                        lines = [line.strip().split("\t") for line in lines]

                        # Write to dest_labels_dir
                        dest_annotation_file = open(os.path.join(dest_labels_dir,d,file), "w")
                        mapping_to_5_classes(lines, dest_annotation_file)
                        dest_annotation_file.close()
                    
                    pbar.update(1)



# Applied to list of lines in .txt file
def mapping_to_5_classes(lines, file):
    new_mapping = {
        "1" : "0",
        "0" : "1",
        "8" : "2",
        "4" : "3"
        # others: 4
    }

    for line in lines:
        # line is a list
        class_id = line[0]
        other_details = line[1:]

        if class_id in new_mapping:
            new_class_id = new_mapping[class_id]
        
        else:
            new_class_id = "4"
        
        line[0] = new_class_id

        file.write(f"{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Convert labels for NTU PCB dataset')
    parser.add_argument("--source_labels_dir",
                        type=str,
                        required=True,
                        help="The path of directory containing NTU PCB labels")
    
    parser.add_argument("--dest_labels_dir",
                        type=str,
                        required=True,
                        help="The path of directory to store converted NTU PCB labels")    

    parser.add_argument('-id',
                        '--images_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where bounding boxed images are stored")
    
    args = parser.parse_args()

    # create directories if not exist
    if not os.path.exists(args.dest_labels_dir):
        os.makedirs(args.dest_labels_dir)

        # Create train val test on split dir
        train_dir = os.path.join(args.dest_labels_dir, 'train')
        val_dir = os.path.join(args.dest_labels_dir, 'val')
        test_dir = os.path.join(args.dest_labels_dir, 'test')

        for dir_path in [train_dir, val_dir, test_dir]:
            os.makedirs(dir_path, exist_ok=True)    

        convert_classes(args.source_labels_dir, args.dest_labels_dir)

    # create directories if      not exist
    # if not os.path.exists(args.images_dest_dir):
    #     os.makedirs(args.images_dest_dir)
    check_yolo_annotations(args.dest_labels_dir, args.dest_labels_dir, args.images_dest_dir)



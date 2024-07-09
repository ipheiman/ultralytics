import os
from tqdm import tqdm

def main():
    det_dir = "/home/heiman/L3/CODE/ultralytics/runs/detect/exp1_fpic_test_SAHI/"
    files = os.listdir(det_dir)

    with tqdm(total = len(files)) as pbar:
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(det_dir,file), "r") as f:
                    lines = f.readlines()

                # Process the lines
                lines = [line.strip().split() for line in lines]

                with open(os.path.join(det_dir,file), "w") as annotation_file:
                    for line in lines:
                        class_id = int(line[0])
                        x1 = float(line[1])
                        y1 = float(line[2])
                        x2 = float(line[3])
                        y2 = float(line[4])    
                        annotation_file.write(f"{class_id} 1.0 {x1} {y1} {x2} {y2}\n")
                    annotation_file.close()
            pbar.update(1)

if __name__ == "__main__":
    main()
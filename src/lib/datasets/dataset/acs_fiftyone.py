import fiftyone as fo
import os
import pandas as pd
import glob

TRAIN_PATH = "/data/shared-data/scalpel/kristina/data/detection/usage/all data/train/basic/"
VAL_PATH = "/data/shared-data/scalpel/kristina/data/detection/usage/all data/test and val/val/"
# Take only files from camera 1
ANN_SELECT_PATTERN = '*_1.txt'

DATASET_NAME_TRAIN = "ACS_usage_train"
DATASET_NAME_VAL = "ACS_usage_val"

index2label = {
    0: "R",
    1: "L",
    2: "R", #"R2",
    3: "L", #"L2",
    4: "R+1",
    5: "L+1",
    6: "R+1", #"R2+1",
    7: "L+1", #"L2+1",
    8: "R+2",
    9: "L+2",
    10: "R+2", #"R2+2",
    11: "L+2", #"L2+2",
    12: "R+3",
    13: "L+3",
    14: "R+3", #"R2+3",
    15: "L+3", #,"L2+3",
    16: "R+4",
    17: "L+4",
    18: "R+4", #"R2+4",
    19: "L+4" #"L2+4"
}

def create_detection(label, x, y, w, h):
    return fo.Detection(label=label,
                        bounding_box=[x,y,w,h])

def create_dataset(name, img_path, ann_path, ann_ptrn):
    images_labels = {}
    for ann_file in glob.glob(os.path.join(ann_path, ann_ptrn)):
        image_file = os.path.join(img_path, os.path.basename(ann_file).replace('.txt','.jpg'))
        # read annotation file
        ann_df = pd.read_csv(ann_file, delimiter=' ', names=['class', 'x', 'y', 'w', 'h'])
        if not os.path.exists(image_file):
            continue
        if image_file not in images_labels:
            images_labels[image_file] = []
        for i, ann in ann_df.iterrows():
            # create a detection label
            label = index2label[int(ann['class'])]
            x = ann['x'] - ann['w']/2
            y = ann['y'] - ann['h']/2
            width = ann['w']
            height = ann['h']
            detection_lbl = create_detection(label=label,
                                             x=x,
                                             y=y,
                                             w=width,
                                             h=height)
            # add the label to dictionary
            images_labels[image_file].append(detection_lbl)

    # create fifty-one dataset
    dataset = fo.Dataset(name)

    for sample_path, detections in images_labels.items():
        sample = fo.Sample(filepath=sample_path,
                           ground_truth=fo.Detections(detections=detections))
        dataset.add_sample(sample)

    dataset.persistent = True
    dataset.save()


if __name__ == '__main__':
    # create train dataset
    create_dataset(name=DATASET_NAME_TRAIN,
                   img_path=os.path.join(TRAIN_PATH, "images"),
                   ann_path=os.path.join(TRAIN_PATH, "labels"),
                   ann_ptrn=ANN_SELECT_PATTERN)
    # create val dataset
    create_dataset(name=DATASET_NAME_VAL,
                   img_path=os.path.join(VAL_PATH, "images"),
                   ann_path=os.path.join(VAL_PATH, "labels"),
                   ann_ptrn=ANN_SELECT_PATTERN)
    exit(0)

from lib.datasets.dataset.acs_fiftyone import DATASET_NAME_VAL, DATASET_NAME_TRAIN
import fiftyone as fo
from fiftyone.types import COCODetectionDataset
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "/data/home/ssaricha/CenterNet/data/acs/"

TRAIN_ANN_FILE = "acs_train_coco.json"
VAL_ANN_FILE = "acs_val_coco.json"

ACS_TRAIN_ANN_PATH = os.path.join(OUTPUT_DIR, "acs_train_coco.json")
ACS_VAL_ANN_PATH = os.path.join(OUTPUT_DIR, "acs_val_coco.json")

def plot_gt_labels_hist(dataset, title=''):
    # Compute a histogram of the predicted labels in the ground_truth (can be also `predictions`) field
    counts = dataset.count_values("ground_truth.detections.label")
    # print(counts)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.bar(counts.keys(), counts.values(), color='g')
    plt.show()


if __name__ == '__main__':
    label_field = "ground_truth"
    dataset_type = COCODetectionDataset

    train_dataset = fo.load_dataset(DATASET_NAME_TRAIN)
    plot_gt_labels_hist(train_dataset, title="Train labels histogram")

    train_dataset.export(
        labels_path=ACS_TRAIN_ANN_PATH,
        dataset_type=dataset_type,
        label_field=label_field,
    )

    val_dataset = fo.load_dataset(DATASET_NAME_VAL)
    plot_gt_labels_hist(val_dataset, title="Val labels histogram")

    val_dataset.export(
        labels_path=ACS_VAL_ANN_PATH,
        dataset_type=dataset_type,
        label_field=label_field,
    )

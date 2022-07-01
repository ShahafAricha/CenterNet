"""
SHAHAF's dataset V1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import os

import src.lib.datasets.dataset.acs_fiftyone

PYCHARM = os.environ['USING_PYCHARM']=="1"
if PYCHARM:
    from src.lib.datasets.dataset.acs_fiftyone import TRAIN_PATH, VAL_PATH, DATASET_NAME_TRAIN, DATASET_NAME_VAL
    from src.lib.datasets.dataset.acs_coco_ann import ACS_TRAIN_ANN_PATH, ACS_VAL_ANN_PATH
    from src.tools.voc_eval_lib.datasets.pascal_voc import pascal_voc
    from src.tools.voc_eval_lib.model.test import apply_nms
else:
    from acs_fiftyone import TRAIN_PATH, VAL_PATH, DATASET_NAME_TRAIN, DATASET_NAME_VAL
    from acs_coco_ann import ACS_TRAIN_ANN_PATH, ACS_VAL_ANN_PATH
# from src.tools.reval import from_dets
import fiftyone as fo
from fiftyone.types import FiftyOneImageDetectionDataset
import cv2

import torch.utils.data as data



class ACS(data.Dataset):
    num_classes = 10 # we unite the 1st and 2nd participants
    default_resolution = [384, 384]
    mean = np.array([0.4281, 0.4367, 0.4946],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.2237, 0.2367, 0.2714],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super().__init__()
        # self.data_dir = os.path.join(opt.data_dir, 'voc')
        _img_path = {'train': TRAIN_PATH, 'val': VAL_PATH}
        self.data_dir = _img_path[split]
        self.img_dir = os.path.join(self.data_dir, 'images')
        _ann_path = {'train': ACS_TRAIN_ANN_PATH, 'val': ACS_VAL_ANN_PATH}
        _ann_name = {'train': 'acs_train_coco', 'val': 'acs_val_coco'}
        _dataset_fo = {'train': DATASET_NAME_TRAIN, 'val': DATASET_NAME_VAL}
        self.dataset_fo = fo.load_dataset(_dataset_fo[split])
        self.annot_path = _ann_path[split]
        self.max_objs = 5
        # NOTE! The order of the classes must be the same as in the exported coco json file (under the "categories" key
        # value). usually it's a lexicographic order
        self.class_name = ['__background__', "L", "L+1", "L+2", "L+3", "L+4", "R", "R+1", "R+2", "R+3", "R+4"]
        self._valid_ids = np.arange(0, 10, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing pascal {} data.'.format(_ann_name[split]))
        self.coco = coco.COCO(self.annot_path)
        self.images = sorted(self.coco.getImgIds())
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = [[[] for __ in range(self.num_samples)] \
                      for _ in range(self.num_classes + 1)]
        for i in range(self.num_samples):
            img_id = self.images[i]
            for j in range(1, self.num_classes + 1):
                if isinstance(all_bboxes[img_id][j], np.ndarray):
                    detections[j][i] = all_bboxes[img_id][j].tolist()
                else:
                    detections[j][i] = all_bboxes[img_id][j]
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):

        factor_w = 1 / self.opt.input_width
        factor_h = 1 / self.opt.input_height

        def create_detection(label, confidence, tl_x, tl_y, br_x, br_y) -> fo.Detection:
            w = max(0, br_x - tl_x)
            h = max(0, br_y - tl_y)
            return fo.Detection(label=label,
                                bounding_box=[tl_x * factor_w,
                                              tl_y * factor_h,
                                              w * factor_w,
                                              h * factor_h],
                                confidence=confidence)

        def create_detections(sample_result:list) -> fo.Detections:
            return fo.Detections(detections=[create_detection(label=self.class_name[int(det[5])+1],
                                                              confidence=det[4],
                                                              tl_x=det[0],
                                                              tl_y=det[1],
                                                              br_x=det[2],
                                                              br_y=det[3]) for det in sample_result])
        # create fifty-one dataset
        dataset = fo.Dataset("results")
        # load images coco meta-data
        images_data = self.coco.loadImgs(ids=results.keys())
        # create temporary fifty-one dataset with the ground-truth and predictions
        for img in images_data:
            image_file = os.path.join(self.img_dir, img['file_name'])
            # get sample from fo dataset
            sample = self.dataset_fo[image_file]
            sample["predictions"] = create_detections(results[img['id']])
            dataset.add_sample(sample)
        # we need to save fo dataset before export
        dataset.save()
        # export the fo dataset with predictions to save_dir path in fo format
        dataset.export(
            dataset_type=FiftyOneImageDetectionDataset,
            labels_path=os.path.join(save_dir, 'fo_exported_results.json'),
            label_field="predictions",
        )



class ACS_STATISTICS_HANDLER(ACS):
    def __init__(self, split, img_files, transform=None):
        super(ACS_STATISTICS_HANDLER, self).__init__(None, split)
        self.transform = transform
        self.img_files = img_files


    def __getitem__(self, idx):
        # import
        path = self.img_files[idx]
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


if __name__ == '__main__':
    """
    THIS script is for calculate dataset statistics like mean and std
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import glob
    image_files = glob.glob(os.path.join(TRAIN_PATH, 'images', "*_1.jpg"))
    # load train dataset
    augs = A.Compose([A.Resize(height=512,
                               width=512),
                      A.Normalize(mean=(0, 0, 0),
                                  std=(1, 1, 1)),
                      ToTensorV2()])
    train_dataset = ACS_STATISTICS_HANDLER('train',img_files=image_files,transform=augs)
    device = torch.device('cpu')
    image_size = 512
    batch_size = 8
    num_workers = 1
    image_loader = data.DataLoader(train_dataset,
                                   batch_size  = batch_size,
                                   shuffle     = False,
                                   num_workers = num_workers,
                                   pin_memory  = True)
    # # display images
    # for batch_idx, inputs in enumerate(image_loader):
    #     fig = plt.figure(figsize=(14, 7))
    #     for i in range(8):
    #         ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    #         plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
    #     break

    ####### COMPUTE MEAN / STD

    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(train_dataset) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))

    exit(0)

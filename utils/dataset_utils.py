import os
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class BokehDataset_ebb(Dataset):
    def __init__(self, args, task='bokeh', train=True):
        super(BokehDataset_ebb, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.train = train

        self.task_dict = {'bokeh': 0}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        self.ids = []
        name_list = os.listdir(self.args.bokeh_path + 'input/')
        print(self.args.bokeh_path)
        self.ids += [self.args.bokeh_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        gt_name = degraded_name.replace("input", "target")
        depth_name = degraded_name.replace("input", "depth")
        depth_name = depth_name.replace("jpg", "png")
        mask_name = degraded_name.replace("input", "focal")
        return gt_name, depth_name, mask_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path, depth_path, mask_path = self._get_gt_path(degraded_path)

        degraded_img = np.array(Image.open(degraded_path).convert('RGB'))
        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        depth_img = np.array(Image.open(depth_path).convert('RGB'))
        mask_img = np.array(Image.open(mask_path).convert('RGB'))
        if self.train:
            crop_size = (1024, 1024)
            x = random.randint(0, degraded_img.shape[1] - crop_size[1])
            y = random.randint(0, degraded_img.shape[0] - crop_size[0])
            degraded_img = degraded_img[y:y + crop_size[0], x:x + crop_size[1]]
            clean_img = clean_img[y:y + crop_size[0], x:x + crop_size[1]]
            depth_img = depth_img[y:y + crop_size[0], x:x + crop_size[1]]
            mask_img = mask_img[y:y + crop_size[0], x:x + crop_size[1]]
        clean_img, degraded_img, depth_img, mask_img = self.toTensor(clean_img), self.toTensor(degraded_img), self.toTensor(depth_img), self.toTensor(mask_img)
        degraded_name = degraded_path.split('/')[-1][:-4]


        return [degraded_name], degraded_img, clean_img, depth_img, mask_img

    def __len__(self):
        return self.length

class BokehDataset_vabd(Dataset):
    def __init__(self, args, task='bokeh', train=True):
        super(BokehDataset_vabd, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.train = train

        self.task_dict = {'bokeh': 0}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        self.ids = []
        name_list = os.listdir(self.args.bokeh_path + 'input/')
        # print(name_list)
        print(self.args.bokeh_path)
        self.ids += [self.args.bokeh_path + 'input/' + id_ for id_ in name_list]
        self.ids *= 10

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name, num):
        # print(num)
        if num==0:
            gt_name = degraded_name.replace("input", "1_8")
            number = torch.FloatTensor([1.8])
        elif num==1:
            gt_name = degraded_name.replace("input", "2_8")
            number = torch.FloatTensor([2.8])
        else:
            gt_name = degraded_name.replace("input", "8")
            number = torch.FloatTensor([8])
        depth_name = degraded_name.replace("input", "depth")
        depth_name = depth_name.replace("JPG", "png")
        depth_name = depth_name.replace("jpg", "png")
        mask_name = degraded_name.replace("input", "focal")
        mask_name = mask_name.replace("JPG", "png")
        mask_name = mask_name.replace("jpg", "png")
        return gt_name, depth_name, mask_name, number

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        # clean_path, depth_path, mask_path, number = self._get_gt_path(degraded_path, random.randint(0, 2))
        clean_path, depth_path, mask_path, number = self._get_gt_path(degraded_path, 1)

        degraded_img = np.array(Image.open(degraded_path).convert('RGB'))
        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        depth_img = np.array(Image.open(depth_path).convert('RGB'))
        mask_img = np.array(Image.open(mask_path).convert('RGB'))
        if self.train:
            crop_size = (1024, 1024)
            x = random.randint(0, degraded_img.shape[1] - crop_size[1])
            y = random.randint(0, degraded_img.shape[0] - crop_size[0])
            degraded_img = degraded_img[y:y + crop_size[0], x:x + crop_size[1]]
            clean_img = clean_img[y:y + crop_size[0], x:x + crop_size[1]]
            depth_img = depth_img[y:y + crop_size[0], x:x + crop_size[1]]
            mask_img = mask_img[y:y + crop_size[0], x:x + crop_size[1]]
        clean_img, degraded_img, depth_img, mask_img = self.toTensor(clean_img), self.toTensor(degraded_img), self.toTensor(depth_img), self.toTensor(mask_img)
        degraded_name = degraded_path.split('/')[-1][:-4]


        return [degraded_name], degraded_img, clean_img, depth_img, mask_img, number

    def __len__(self):
        return self.length



class BokehDataset_test_vabd(Dataset):
    def __init__(self, args, task='bokeh', train=True, num=0):
        super(BokehDataset_test_vabd, self).__init__()
        self.ids = []
        self.num=num
        self.task_idx = 0
        self.args = args
        self.train = train

        self.task_dict = {'bokeh': 0}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        self.ids = []
        name_list = os.listdir(self.args.bokeh_path + 'input/')
        # print(name_list)
        print(self.args.bokeh_path)
        self.ids += [self.args.bokeh_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.num==0:
            gt_name = degraded_name.replace("input", "1_8")
            number = torch.FloatTensor([1.8])
        elif self.num==1:
            gt_name = degraded_name.replace("input", "2_8")
            number = torch.FloatTensor([2.8])
        else:
            gt_name = degraded_name.replace("input", "8")
            number = torch.FloatTensor([8])
        depth_name = degraded_name.replace("input", "depth")
        depth_name = depth_name.replace("JPG", "png")
        depth_name = depth_name.replace("jpg", "png")
        mask_name = degraded_name.replace("input", "tidumask")
        mask_name = mask_name.replace("JPG", "png")
        mask_name = mask_name.replace("jpg", "png")
        return gt_name, depth_name, mask_name, number

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path, depth_path, mask_path, number = self._get_gt_path(degraded_path)

        degraded_img = np.array(Image.open(degraded_path).convert('RGB'))
        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        depth_img = np.array(Image.open(depth_path).convert('RGB'))
        mask_img = np.array(Image.open(mask_path).convert('RGB'))
        if self.train:
            crop_size = (1024, 1024)
            x = random.randint(0, degraded_img.shape[1] - crop_size[1])
            y = random.randint(0, degraded_img.shape[0] - crop_size[0])
            degraded_img = degraded_img[y:y + crop_size[0], x:x + crop_size[1]]
            clean_img = clean_img[y:y + crop_size[0], x:x + crop_size[1]]
            depth_img = depth_img[y:y + crop_size[0], x:x + crop_size[1]]
            mask_img = mask_img[y:y + crop_size[0], x:x + crop_size[1]]
        clean_img, degraded_img, depth_img, mask_img = self.toTensor(clean_img), self.toTensor(degraded_img), self.toTensor(depth_img), self.toTensor(mask_img)
        degraded_name = degraded_path.split('/')[-1][:-4]


        return [degraded_name], degraded_img, clean_img, depth_img, mask_img, number

    def __len__(self):
        return self.length
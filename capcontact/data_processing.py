import torch
from torch.utils.data.dataset import Dataset

import os
import glob
import pandas as pd
import numpy as np
import skimage.measure
import scipy.stats as st

from capcontact.config import TOUCHSENSOR_CONFIG, LOSS_WEIGHT_CONFIG


class CapFTIRDataset(Dataset):
    
    def __init__(self, path, flip=True):
        super(CapFTIRDataset, self).__init__()

        if type(path) == list:
            self.filenames = path
        elif os.path.isdir(path):
            self.filenames = glob.glob(os.path.join(path, '*.npz'))
        elif path.endswith(".csv"):
            self.filenames = (pd.read_csv(path)["datapath"]).values.tolist()
        else:
            raise NotImplementedError("Please provide a path to a folder, or a list or a csv file containing the paths to the respective files.")
        
        self.flip = flip

    def __getitem__(self, index):
        fileindex = os.path.basename(self.filenames[index]).split(".")[0]
        dirname = os.path.dirname(self.filenames[index])
        participant, block = dirname.split("/")[-3:-1]
        fileinfo = (participant, block, fileindex)
        datapair = np.load(self.filenames[index])

        cap = datapair["cap"].astype(np.float32)/TOUCHSENSOR_CONFIG.MAX_VALUE
        cap[cap < 0] = 0

        ftir = 1 - datapair["ftir"].astype(np.float32)/255
        ftir_touches = get_FTIR_touches(ftir)

        total_area = ftir.shape[0]*ftir.shape[1]
        touch_area = 0
        touch_count = 0
        weights = np.zeros(ftir.shape)

        for ftir_touch in ftir_touches:
            col_span = ftir_touch["bbox_col_max"] - ftir_touch["bbox_col_min"]
            row_span = ftir_touch["bbox_row_max"] - ftir_touch["bbox_row_min"]
            col_centre = int(ftir_touch["bbox_col_min"] + 0.5 * col_span)
            row_centre = int(ftir_touch["bbox_row_min"] + 0.5 * row_span)
            bbox_col_min = max(0, col_centre - int(0.5*col_span*LOSS_WEIGHT_CONFIG.RECTANGLE_SIZE))
            bbox_row_min = max(0, row_centre - int(0.5*row_span*LOSS_WEIGHT_CONFIG.RECTANGLE_SIZE))
            bbox_col_max = min(ftir.shape[1], col_centre + int(0.5*col_span*LOSS_WEIGHT_CONFIG.RECTANGLE_SIZE))
            bbox_row_max = min(ftir.shape[0], row_centre + int(0.5*col_span*LOSS_WEIGHT_CONFIG.RECTANGLE_SIZE))
            current_touch_area = (bbox_row_max-bbox_row_min)*(bbox_col_max-bbox_col_min)
            touch_area += current_touch_area
            touch_count += 1
            weights[bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max] += get_gaussian_kernel(kernel_len=(bbox_row_max-bbox_row_min, bbox_col_max-bbox_col_min))*current_touch_area

        weights = (1+LOSS_WEIGHT_CONFIG.ALPHA_FACTOR*weights)
        weights = (weights/np.sum(weights))*total_area

        weight_tensor = torch.tensor(weights, requires_grad=False).unsqueeze(0)
        cap_tensor = torch.from_numpy(cap).unsqueeze(0)
        ftir_tensor = torch.from_numpy(ftir).unsqueeze(0)

        if self.flip is True:
            flip = np.random.randint(0, 4)
        else:
            flip = 0
        if flip == 1:
            weight_tensor = torch.flip(weight_tensor, [2])
            cap_tensor = torch.flip(cap_tensor, [2])
            ftir_tensor = torch.flip(ftir_tensor, [2])
        elif flip == 2:
            weight_tensor = torch.flip(weight_tensor, [1])
            cap_tensor = torch.flip(cap_tensor, [1])
            ftir_tensor = torch.flip(ftir_tensor, [1])
        elif flip == 3:
            weight_tensor = torch.flip(weight_tensor, [1, 2])
            cap_tensor = torch.flip(cap_tensor, [1, 2])
            ftir_tensor = torch.flip(ftir_tensor, [1, 2])
        return cap_tensor, ftir_tensor, weight_tensor, flip, fileinfo

    def __len__(self):
        return len(self.filenames)


def get_FTIR_touches(ftir):
    labeled_ftir = skimage.measure.label(ftir, background=0)
    regionprops_ftir = skimage.measure.regionprops(labeled_ftir, intensity_image=ftir)

    ftir_touches = [{"idx": i,
                     "bbox_col_min": regionprops_ftir[i].bbox[1],
                     "bbox_row_min": regionprops_ftir[i].bbox[0],
                     "bbox_col_max": regionprops_ftir[i].bbox[3],
                     "bbox_row_max": regionprops_ftir[i].bbox[2],
                     "area": regionprops_ftir[i].area,
                     "centroid": regionprops_ftir[i].centroid
                     } for i in range(len(regionprops_ftir))]

    return ftir_touches


def get_gaussian_kernel(kernel_len, nsig=(2, 2)):
    x = np.linspace(-nsig[0], nsig[0], kernel_len[0]+1)
    y = np.linspace(-nsig[1], nsig[1], kernel_len[1]+1)
    kernel_x = np.diff(st.norm.cdf(x))
    kernel_y = np.diff(st.norm.cdf(y))

    kernel = np.outer(kernel_x, kernel_y)

    return kernel/kernel.sum()

import torch
from torch.utils.data import DataLoader

import numpy as np
import argparse
import sys
import os
import glob
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from capcontact.config import INFERENCE_CONFIG, TOUCHSENSOR_CONFIG
from capcontact.data_processing import CapFTIRDataset
from capcontact.model import Generator
from capcontact.loss import get_iou, get_mse
from capcontact.visualization import get_bicubic_image


def main(argv):
    parser = argparse.ArgumentParser(description='Inference settings for CapContact.')
    parser.add_argument('--batch_size', default=8, type=int, help='Inference batch size.')
    parser.add_argument('--load_ckpt_path', type=str, required=True, help='Folder with stored models.')
    parser.add_argument('--load_ckpt_epoch', default=-1, type=int, help='Epoch when model was stored.')
    parser.add_argument('--test_set', type=str, required=True, help='Path to folder with test data or path to csv file specifying test data.')
    parser.add_argument('--out_path', default="", type=str, help='Path to folder for saving inference output.')
    parser.add_argument('--skip_saving_images', action='store_false', dest="save_images", help='Set flag when inferred images should not be saved.')

    opt = parser.parse_args(argv[1:])

    BATCH_SIZE = opt.batch_size
    LOAD_CKPT_PATH = opt.load_ckpt_path
    LOAD_CKPT_EPOCH = opt.load_ckpt_epoch
    TEST_SET_PATH = opt.test_set
    SAVE_IMAGES = opt.save_images

    generator = Generator()
    if torch.cuda.is_available():
        generator.cuda()
    DEVICE = next(generator.parameters()).device

    if LOAD_CKPT_EPOCH == -1:
        ckpts = glob.glob(os.path.join(LOAD_CKPT_PATH, "ckpt_epoch_*.tar"))
        ckpt_epochs = [int(os.path.basename(p).replace(".tar", "").replace("ckpt_epoch_", "")) for p in ckpts]
        LOAD_CKPT_EPOCH = np.amax(ckpt_epochs)

    full_load_path = os.path.join(LOAD_CKPT_PATH, "ckpt_epoch_{}.tar".format(LOAD_CKPT_EPOCH))
    checkpoint = torch.load(full_load_path, map_location=DEVICE)
    print(">>> Load network from {}.".format(full_load_path))
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    if opt.out_path == "":
        OUT_PATH = f"inference_output/"
    else:
        OUT_PATH = f"{opt.out_path}/"
    OUT_PATH = os.path.join(OUT_PATH, f"{LOAD_CKPT_PATH.strip('/').split('/')[-1]}_epoch{LOAD_CKPT_EPOCH}")
    
    if not os.path.exists(os.path.join(OUT_PATH, "npz")):
        os.makedirs(os.path.join(OUT_PATH, "npz"))

    if SAVE_IMAGES and not os.path.exists(os.path.join(OUT_PATH, "png")):
        os.makedirs(os.path.join(OUT_PATH, "png"))

    # load datasets
    test_set = CapFTIRDataset(TEST_SET_PATH)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    test_results = {'samples': 0,
                    'iou': 0,
                    'mse': 0
                    }

    csv_fieldnames = ["participant", "block", "index", "IOU", "MSE"]
    f_csv = open(os.path.join(OUT_PATH, "stats.csv"), 'w')
    csv_writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='[Complete inference on test set.]')

        for test_cap, test_ftir, _, _, fileinfo in test_bar:
            test_batch_size = test_cap.size(0)
            test_cap = test_cap.to(DEVICE)
            test_ftir = test_ftir.to(DEVICE)

            test_pred_contact = generator(test_cap)

            iou = get_iou(test_ftir, test_pred_contact, threshold_value=INFERENCE_CONFIG.THRESHOLD)
            test_results['iou'] += iou.sum().item()
            mse = get_mse(test_ftir, test_pred_contact, threshold_value=INFERENCE_CONFIG.THRESHOLD)
            test_results['mse'] += mse.sum().item()
            test_results['samples'] += test_batch_size

            test_cap_denormalized = test_cap * TOUCHSENSOR_CONFIG.MAX_VALUE
            test_pred_contact_binarized = torch.where(test_pred_contact > INFERENCE_CONFIG.THRESHOLD,
                                                      torch.ones_like(test_pred_contact),
                                                      torch.zeros_like(test_pred_contact))

            for i in range(test_ftir.size(0)):
                cap_denormalized = test_cap_denormalized[i].data.cpu().squeeze(0).numpy()
                bicubic = get_bicubic_image(test_cap[i].data.cpu()).squeeze(0).numpy() * TOUCHSENSOR_CONFIG.MAX_VALUE
                pred_contact = test_pred_contact[i].data.cpu().squeeze(0).numpy()
                pred_contact_binarized = test_pred_contact_binarized[i].data.cpu().squeeze(0).numpy()
                ftir = test_ftir[i].data.cpu().squeeze(0).numpy()

                if SAVE_IMAGES:
                    plt.imsave(os.path.join(OUT_PATH, "png", f'out_{fileinfo[0][i]}_{fileinfo[1][i]}_{fileinfo[2][i]}.png'),
                               pred_contact_binarized, vmin=0, vmax=1, cmap="gray")
                
                np.savez_compressed(os.path.join(OUT_PATH, "npz", f"out_{fileinfo[0][i]}_{fileinfo[1][i]}_{fileinfo[2][i]}.npz"),
                                    cap=cap_denormalized,
                                    bicubic=bicubic, 
                                    ftir=ftir, 
                                    pred_contact=pred_contact,
                                    pred_contact_binarized=pred_contact_binarized)

                csv_writer.writerow({
                    "participant": fileinfo[0][i],
                    "block": fileinfo[1][i],
                    "index": fileinfo[2][i],
                    "IOU": iou[i].item(),
                    "MSE": mse[i].item(),
                    }
                    )

    print("RESULTS:")
    print(f"Mean IOU: {test_results['iou'] / test_results['samples']:.3f}")
    print(f"Mean MSE: {test_results['mse'] / test_results['samples']:.5f}")
    f_csv.close()


if __name__ == '__main__':
    main(sys.argv)

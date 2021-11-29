import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils

import numpy as np
import argparse
import sys
import os
from datetime import datetime
import glob
from tqdm import tqdm

from capcontact.config import TRAINING_CONFIG
from capcontact.data_processing import CapFTIRDataset
from capcontact.model import Critic, Generator
from capcontact.loss import get_critic_loss, weighted_mse_loss, get_iou, get_mse
from capcontact.visualization import get_bicubic_image


def main(argv):
    parser = argparse.ArgumentParser(description='Training settings for CapContact.')
    parser.add_argument('--num_epochs', default=-1, type=int, help='Number of epochs to train the model for. Default as described in paper.')
    parser.add_argument('--batch_size', default=8, type=int, help='Training batch size.')
    parser.add_argument('--load_ckpt_path', default="", type=str, help='Folder with stored models.')
    parser.add_argument('--load_ckpt_epoch', default=-1, type=int, help='Epoch when model was stored.')
    parser.add_argument('--train_set', type=str, required=True, help='Path to folder with training data or path to csv file specifying training data.')
    parser.add_argument('--val_set', type=str, required=True, help='Path to folder with validation data or path to csv file specifying validation data.')
    parser.add_argument('--save_ckpt_path', default="", type=str, help='Path to folder for saving models.')

    opt = parser.parse_args(argv[1:])

    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    LOAD_CKPT_PATH = opt.load_ckpt_path
    LOAD_CKPT_EPOCH = opt.load_ckpt_epoch
    TRAIN_SET_PATH = opt.train_set
    VAL_SET_PATH = opt.val_set

    if opt.save_ckpt_path == "":
        SAVE_PATH = f"saved_models/cktps_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        SAVE_PATH = f"{opt.save_ckpt_path}/cktps_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    tensorboard_writer = SummaryWriter(SAVE_PATH)

    # load datasets
    training_set = CapFTIRDataset(TRAIN_SET_PATH)
    validation_set = CapFTIRDataset(VAL_SET_PATH)
    train_loader = DataLoader(dataset=training_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=validation_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    generator = Generator()
    critic = Critic()

    cap_tensor, ftir_tensor, weight_tensor, _, _ = next(iter(train_loader))
    tensorboard_writer.add_graph(generator, cap_tensor)
    tensorboard_writer.add_graph(critic, ftir_tensor)

    if torch.cuda.is_available():
        generator.cuda()
        critic.cuda()

    DEVICE = next(generator.parameters()).device

    gen_optimizer = optim.Adam(generator.parameters(), lr=TRAINING_CONFIG.INITIAL_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=TRAINING_CONFIG.INITIAL_LR)

    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, 
                                                         NUM_EPOCHS if NUM_EPOCHS!=-1 else TRAINING_CONFIG.DEFAULT_EPOCH, 
                                                         eta_min=TRAINING_CONFIG.MINIMUM_LR)
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, 
                                                            NUM_EPOCHS if NUM_EPOCHS!=-1 else TRAINING_CONFIG.DEFAULT_EPOCH, 
                                                            eta_min=TRAINING_CONFIG.MINIMUM_LR)

    start_epoch = 0

    if LOAD_CKPT_PATH != "":
        if LOAD_CKPT_EPOCH == -1:
            ckpts = glob.glob(os.path.join(LOAD_CKPT_PATH,"ckpt_epoch_*.tar"))
            ckpt_epochs = [int(os.path.basename(p).replace(".tar","").replace("ckpt_epoch_", "")) for p in ckpts]
            LOAD_CKPT_EPOCH = np.amax(ckpt_epochs)

        full_load_path = os.path.join(LOAD_CKPT_PATH, "ckpt_epoch_{}.tar".format(LOAD_CKPT_EPOCH))
        checkpoint = torch.load(full_load_path, map_location=DEVICE)
        print(">>> Load network from {}.".format(full_load_path))

        generator.load_state_dict(checkpoint["generator_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        gen_scheduler.load_state_dict(checkpoint["gen_scheduler_state_dict"])
        critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])

        start_epoch = LOAD_CKPT_EPOCH

        print(">>> Resume training from epoch {}.".format(start_epoch))

    # PRETRAINING
    if start_epoch <= 0:
        print(">>> Conduct pretraining with MSE loss!")
        generator.train()
        pretraining_bar = tqdm(train_loader)
        for train_cap, train_ftir, train_weight, _, _ in pretraining_bar:
            gen_optimizer.zero_grad()

            train_cap = train_cap.to(DEVICE)
            train_ftir = train_ftir.to(DEVICE)
            train_weight = train_weight.to(DEVICE)

            train_pred_contact = generator(train_cap)
            generator_loss = weighted_mse_loss(train_pred_contact, train_ftir, train_weight)
            generator_loss.backward()
            gen_optimizer.step()

        generator.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_images = []

            for val_cap, val_ftir, val_weight, _, _ in val_bar:

                val_cap = val_cap.to(DEVICE)
                val_ftir = val_ftir.to(DEVICE)
                val_weight = val_weight.to(DEVICE)

                val_pred_contact = generator(val_cap)

                for i in range(val_ftir.size(0)):
                    if len(val_images) < TRAINING_CONFIG.N_VAL_SNAPSHOTS:
                        val_images.append(
                            [get_bicubic_image(val_cap[i].data.cpu()),
                             val_pred_contact[i].data.cpu(),
                             val_ftir[i].data.cpu()])

            val_save_bar = tqdm(val_images, desc='[Saving pretraining results.]')
            
            index = 0
            for image in val_save_bar:
                image = torchvision.utils.make_grid(image, nrow=3, padding=5, pad_value=1)
                tensorboard_writer.add_image('pretrain/val/index_{}.png'.format(index), image, 1)
                index += 0
    
    # MAIN TRAINING LOOP
    epoch = start_epoch
    best_epoch = epoch
    best_IOU = -np.inf
    print(">>> Start training!")
    while (NUM_EPOCHS == -1 and (epoch < TRAINING_CONFIG.DEFAULT_EPOCH 
           or epoch <= best_epoch+TRAINING_CONFIG.PATIENT_EPOCH)) or \
          (NUM_EPOCHS != -1 and epoch < start_epoch+NUM_EPOCHS):

        # TRAINING
        train_results = {'critic_samples': 0,  
                         'critic_loss': 0,
                         'generator_samples': 0,
                         'generator_loss': 0, 
                         'generator_score': 0, 
                         'generator_mse': 0,
                         'iou': 0,
                         }

        generator.train()
        critic.train()

        train_bar = tqdm(train_loader)
        train_images = []
        
        step = 0
        for train_cap, train_ftir, train_weight, _, _ in train_bar:
            batch_size = train_cap.size(0)

            critic_optimizer.zero_grad()

            train_cap = train_cap.to(DEVICE)
            train_ftir = train_ftir.to(DEVICE)
            train_weight = train_weight.to(DEVICE)

            train_pred_contact = generator(train_cap)
            
            critic_loss = get_critic_loss(train_ftir, train_pred_contact.detach(), critic)

            critic_loss.backward()
            critic_optimizer.step()

            train_results['critic_loss'] += critic_loss.item() * batch_size
            train_results['critic_samples'] += batch_size

            if ((step+1) % TRAINING_CONFIG.CRITIC_ITER) == 0:
                gen_optimizer.zero_grad()

                mse_loss = weighted_mse_loss(train_ftir, train_pred_contact, train_weight)
                pred_contact_score = critic(train_pred_contact)
                adversarial_loss = -pred_contact_score.mean()

                generator_loss = mse_loss + adversarial_loss
                generator_loss.backward()
                gen_optimizer.step()

                train_results['generator_loss'] += generator_loss.item() * batch_size
                train_results['generator_mse'] += mse_loss.item() * batch_size
                train_results['generator_score'] += pred_contact_score.sum().item()
                train_results['generator_samples'] += batch_size

                with torch.no_grad():
                    iou = get_iou(train_ftir, train_pred_contact, threshold_value=TRAINING_CONFIG.THRESHOLD)
                train_results['iou'] += iou.sum().item()

                if len(train_images) < TRAINING_CONFIG.N_VAL_SNAPSHOTS:
                    for i in range(min(TRAINING_CONFIG.N_VAL_SNAPSHOTS-len(train_images), batch_size)):
                        train_images.append(
                                            [get_bicubic_image(train_cap[i].data.cpu()),
                                             train_pred_contact[i].data.cpu(),
                                             train_ftir[i].data.cpu()])

                train_bar.set_description(
                    desc="[Training] Epoch: {:d} — Generator loss: {:.4f} —  Generator score: {:.4f}  — Critic loss: {:.4f} — IOU: {:.4f}".format(
                            epoch,
                            train_results['generator_loss'] / train_results['generator_samples'],
                            train_results['generator_score'] / train_results['generator_samples'],
                            train_results['critic_loss'] / train_results['critic_samples'],
                            train_results['iou'] / train_results['generator_samples'])
                )

            step += 1

        train_save_bar = tqdm(train_images, desc='[Saving training images]')
        index = 0
        for image in train_save_bar:
            image = torchvision.utils.make_grid(image, nrow=3, padding=5, pad_value=1)
            tensorboard_writer.add_image('train/train/index_{}.png'.format(index), image, epoch)
            index += 1

        # VALIDATION
        generator.eval()
        critic.eval()

        val_results = {'samples': 0,
                       'iou': 0,
                       'mse': 0
                       }

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_images = []

            for val_cap, val_ftir, val_weight, _, _ in val_bar:
                val_batch_size = val_cap.size(0)

                val_cap = val_cap.to(DEVICE)
                val_ftir = val_ftir.to(DEVICE)
                val_weight = val_weight.to(DEVICE)

                val_pred_contact = generator(val_cap)

                for i in range(val_ftir.size(0)):
                    if len(val_images) < TRAINING_CONFIG.N_VAL_SNAPSHOTS:
                        val_images.append(
                            [get_bicubic_image(val_cap[i].data.cpu()),
                             val_pred_contact[i].data.cpu(),
                             val_ftir[i].data.cpu()])

                iou = get_iou(val_ftir, val_pred_contact, threshold_value=TRAINING_CONFIG.THRESHOLD)
                val_results['iou'] += iou.sum().item()
                mse = get_mse(val_ftir, val_pred_contact, threshold_value=TRAINING_CONFIG.THRESHOLD)
                val_results['mse'] += mse.sum().item()
                val_results['samples'] += val_batch_size

                val_bar.set_description(
                    desc='[Validation] Epoch: {:d} — IOU: {:.4f} — MSE: {:.4f}'.format(
                        epoch, 
                        val_results['iou']/val_results['samples'], 
                        val_results['mse']/val_results['samples']))

                if len(val_images) < TRAINING_CONFIG.N_VAL_SNAPSHOTS:
                    for i in range(min(TRAINING_CONFIG.N_VAL_SNAPSHOTS-len(val_images), val_batch_size)):
                        val_images.append(
                                          [get_bicubic_image(val_cap[i].data.cpu()),
                                           val_pred_contact[i].data.cpu(),
                                           val_ftir[i].data.cpu()])

            val_save_bar = tqdm(val_images, desc='[Saving validation images.]')
            index = 0
            for image in val_save_bar:
                image = torchvision.utils.make_grid(image, nrow=3, padding=5, pad_value=1)
                tensorboard_writer.add_image('train/val/index_{}.png'.format(index), image, epoch)
                index += 1

        if (val_results['iou'] / val_results['samples']) > best_IOU:
            best_IOU = val_results['iou'] / val_results['samples']
            best_epoch = epoch

        # save model parameters
        torch.save({"generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                    "critic_optimizer_state_dict": critic_optimizer.state_dict(),
                    "gen_scheduler_state_dict": gen_scheduler.state_dict(),
                    "critic_scheduler_state_dict": critic_scheduler.state_dict(),
                    "epoch": epoch,
                    'train/critic_loss': train_results['critic_loss'] / train_results['critic_samples'],
                    'train/generator_loss': train_results['generator_loss'] / train_results['generator_samples'],
                    'train/generator_score': train_results['generator_score'] / train_results['generator_samples'],
                    'train/generator_mse': train_results['generator_mse'] / train_results['generator_samples'],
                    'train/IOU': train_results['iou'] / train_results['generator_samples'],
                    'val/MSE': val_results['mse'] / val_results['samples'],
                    'val/IOU': val_results['iou'] / val_results['samples'],
                    }, os.path.join(SAVE_PATH, "ckpt_epoch_{}.tar".format(epoch)))

        tensorboard_writer.add_scalar('train/train/critic_loss', train_results['critic_loss'] / train_results['critic_samples'], epoch)
        tensorboard_writer.add_scalar('train/train/generator_loss', train_results['generator_loss'] / train_results['generator_samples'], epoch)
        tensorboard_writer.add_scalar('train/train/generator_score', train_results['generator_score'] / train_results['generator_samples'], epoch)
        tensorboard_writer.add_scalar('train/train/generator_mse', train_results['generator_mse'] / train_results['generator_samples'], epoch)
        tensorboard_writer.add_scalar('train/train/IOU', train_results['iou'] / train_results['generator_samples'], epoch)
        tensorboard_writer.add_scalar('train/val/MSE', val_results['mse'] / val_results['samples'], epoch)
        tensorboard_writer.add_scalar('train/val/IOU', val_results['iou'] / val_results['samples'], epoch)

        gen_scheduler.step()
        critic_scheduler.step()

        epoch += 1

    tensorboard_writer.close()   


if __name__ == '__main__':
    main(sys.argv)

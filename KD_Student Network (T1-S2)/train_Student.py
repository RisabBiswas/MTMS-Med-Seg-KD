import os
import time
from glob import glob
import torch.optim as optim 
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import SpleenDataset
from model import build_small_unet
from model_T import MultiTask_U_Net
from loss import DiceLoss, DiceBCELoss, PredictionMapDistillation
from utils import seeding, create_dir, epoch_time
import torch.nn as nn
from info_nce import InfoNCE
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model_SNet, model_TNet, loader, optimizer, loss_fn):
    epoch_loss = 0.0
    model_SNet.train()
    model_TNet.eval()  # Set the teacher model to evaluation mode

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        bottleneck_Feats_S, p2_S, d2_S, y_pred = model_SNet(x)

        with torch.no_grad():
            bottleneck_Feats_T, p2_T, d4_T, segmentation_output, _ = model_TNet(x)

        seg_loss = loss_fn(y_pred, y)  # segmentation loss
        bottleneck_Feats_T_resized = F.interpolate(bottleneck_Feats_T, size=(64, 64), mode='bilinear', align_corners=False)

        conv_reduce_channels = nn.Conv2d(1024, 128, kernel_size=1).to(device)
        bottleneck_Feats_T_reduced = conv_reduce_channels(bottleneck_Feats_T_resized).to(device)

        # Calculate bottleneck to bottleneck contrastive loss
        info_nce_loss = InfoNCE(temperature=0.1)
        bottleneck_Feats_S = bottleneck_Feats_S.view(batch_size, -1)
        bottleneck_Feats_T_reduced = bottleneck_Feats_T_reduced.view(batch_size, -1)
        b_b_Con_loss = info_nce_loss(bottleneck_Feats_S, bottleneck_Feats_T_reduced)

        # Calculate ecoder to encoder contrastive loss
        conv_reduce_channels2 = nn.Conv2d(128, 64, kernel_size=1).to(device)
        conv_reduce_channels3 = nn.Conv2d(32, 64, kernel_size=1).to(device)
        p2_T = conv_reduce_channels2(p2_T).to(device)
        p2_S = p2_S.view(batch_size, -1)
        p2_T = p2_T.view(batch_size, -1)
        loss_P2 = info_nce_loss(p2_S, p2_T)
        #loss_P2 = nn.MSELoss()(p2_S, p2_T_reduced)

        d2_S = conv_reduce_channels3(d2_S).to(device)
        d2_S = d2_S.view(batch_size, -1)
        d4_T = d4_T.view(batch_size, -1)
        loss_decoder = info_nce_loss(d2_S, d4_T)

        prediction_map_distillation = PredictionMapDistillation()
        loss_pmd = prediction_map_distillation(y_pred, segmentation_output)
        seg_loss_weight = 0.4  # Adjust these weights as needed
        b_b_Con_loss_weight = 0.2
        pmd_loss_weight = 0.2
        e_e_con_loss_weight = 0.1
        d_d_con_loss_weight = 0.1
        loss = seg_loss * seg_loss_weight + d_d_con_loss_weight * loss_decoder + b_b_Con_loss_weight * b_b_Con_loss + pmd_loss_weight * loss_pmd + e_e_con_loss_weight * loss_P2

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss

def evaluate(model_SNet, loader, loss_fn):
    epoch_loss = 0.0

    model_SNet.eval()  # Only set student model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            _, _, _, y_pred = model_SNet(x)  # Obtain student predictions

            seg_loss = loss_fn(y_pred, y)  # Calculate segmentation loss only
            epoch_loss += seg_loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("Modelfiles_KD_B-B+E-E+D-D+PMD")

    """ Load dataset """
    train_x = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student\train_data\image\*'))
    train_y = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student\train_data\mask\*'))

    valid_x = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student\test_data\image\*'))
    valid_y = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student\test_data\mask\*'))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 256
    W = 256
    size = (H, W)
    batch_size = 20
    num_epochs = 120
    lr = 1e-2
    checkpoint_path = r'D:\Risab\Thesis_Experiment_v2\Spleen_KD_Student\Modelfiles_KD_B-B+E-E+D-D+PMD\KD_B-B+E-E+D-D+PMD_Checkpoint.pth'

    """ Dataset and loader """
    train_dataset = SpleenDataset(train_x, train_y)
    valid_dataset = SpleenDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    """Load Features from Multi Task Teacher UNET"""
    model_TNet = MultiTask_U_Net(num_classes_segmentation=1, num_channels_reconstruction=3).to(device)
    checkpoint_dir = r'D:\Risab\Medical_Experiments\Spleen_KD_Student\Multitask_FreezedModel'
    checkpoint_path_T = os.path.join(checkpoint_dir, 'MultiTask_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path_T, map_location=device)
    model_TNet.load_state_dict(checkpoint)
    model_TNet.eval()

    """Define Student UNET"""
    model_SNet = build_small_unet()
    model_SNet = model_SNet.to(device)

    #optimizer = optim.AdamW(model_SNet.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)
    # optimizer = optim.Adagrad(model_SNet.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    optimizer = optim.RMSprop(model_SNet.parameters(), lr=lr, alpha=0.99, eps=1e-08) 

    """ Training the model """
    best_valid_loss = float("inf")
    loss_fn = DiceLoss()
    train_segmentation_losses = []
    valid_segmentation_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model_SNet, model_TNet, train_loader, optimizer, loss_fn)
        valid_loss = evaluate(model_SNet, valid_loader, loss_fn)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model_SNet.state_dict(), checkpoint_path)

        #scheduler.step(valid_loss)  # Update learning rate scheduler

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

        train_segmentation_losses.append(train_loss)
        valid_segmentation_losses.append(valid_loss)

        # Plot the training curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), train_segmentation_losses, label='Train Segmentation Loss')
        plt.plot(range(1, epoch+2), valid_segmentation_losses, label='Valid Segmentation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.savefig('Training_Curve_KD_E-E+PMD.png')
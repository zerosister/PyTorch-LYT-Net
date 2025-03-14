from ast import arg
from sched import scheduler
import debugpy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from model import LYT
from losses import CombinedLoss
from dataloader import create_dataloaders
import os
import numpy as np
from tqdm import tqdm
import logging
import argparse

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
    
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim


    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def setup_logger(log_path='training.log'):
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_path,  # 日志保存到文件
        filemode='w'             # 覆盖模式
    )
    return logging.getLogger()

def main(args):
    # Hyperparameters
    dataset = args.dataset
    dataset_paths = {
        "LOLv1": "LOLv1",
        "LOLv2Real": "LOLv2/Real_captured",
        "LOLv2Synthetic": "LOLv2/Synthetic"
    }

    data_path = dataset_paths.get(dataset, "LOLv1")
    train_low = f'data/{data_path}/Train/Low'
    train_high = f'data/{data_path}/Train/Normal'
    test_low = f'data/{data_path}/Test/Low'
    test_high = f'data/{data_path}/Test/Normal'
    
    lab_name = args.lab_name
    folder = os.path.join(os.getcwd(),"experiments", lab_name)
    os.makedirs(folder, exist_ok=True)
    model_dir = os.path.join(folder, "models")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Made lab Dir {folder} and {model_dir}")
    log_path = os.path.join(folder, "train.log")
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"using cuda:{args.cuda}")
    logger = setup_logger(log_path)
    print(f"output log in {log_path}")
    
    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    logger.info(f"Train loader: {len(train_loader)}; Test loader: {len(test_loader)}")

    # learning rate setting 
    learning_rate = 2e-4 
    min_lr = 1e-6
    num_epochs = 1500
    first_decay_steps = 150
    logger.info(f'LR: {learning_rate}; Epochs: {num_epochs}; Decay_steps: {first_decay_steps}')

    
    # Model, loss, optimizer, and scheduler
    model = LYT(args=args).to(device)
    logger.info(model)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=first_decay_steps,T_mult=1,eta_min=min_lr)
    
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    logger.info('Training started.')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True
        )
        for batch_idx, batch in progress_bar:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{train_loss/(batch_idx+1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, LR: {optimizer.param_groups[0]["lr"]:.3e}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(model_dir, "epoch"+str(epoch)+'.pth'))
            logger.info(f'Saving model with PSNR: {best_psnr:.6f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--use_hvi', type=bool, default=False)
    parser.add_argument('--lab_name', type=str, default="expr_1")
    parser.add_argument('--dataset', type=str, default="LOLv1")
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    main(args)
"""
Training script for Swin Transformer Super-Resolution
"""

import os
import sys
import time
import random
import argparse
from tqdm import tqdm
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from swin_sr import SwinSR
from dataset import TurbulenceSRDataset


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictObject:
    """Convert dictionary to object with attribute access"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictObject(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self):
        """Convert DictObject to plain dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictObject):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten the nested structure for easier access
    config = DictObject({})
    
    # Data config
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            setattr(config, key, value)
    
    # Model config
    if 'model' in config_dict:
        config.model = config_dict['model']
    
    # Training config
    if 'training' in config_dict:
        for key, value in config_dict['training'].items():
            setattr(config, key, value)
    
    # Loss config
    if 'loss' in config_dict:
        for key, value in config_dict['loss'].items():
            setattr(config, key, value)
    
    # Logging config
    if 'logging' in config_dict:
        for key, value in config_dict['logging'].items():
            setattr(config, key, value)
    
    # Reproducibility config
    if 'reproducibility' in config_dict:
        for key, value in config_dict['reproducibility'].items():
            setattr(config, key, value)
    
    # Device config
    if 'device' in config_dict:
        config.device = config_dict['device']
    
    return config


def display_config(config):
    """Display configuration"""
    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    print("=" * 80)


def set_seed(seed, deterministic=False):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_loss_function(loss_type):
    """Get loss function based on configuration"""
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_scheduler(optimizer, scheduler_type, config):
    """Get learning rate scheduler"""
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.T_max,
            eta_min=config.eta_min
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.patience,
            factor=config.factor,
            verbose=True
        )
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=getattr(config, 'gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class WarmupScheduler:
    """Learning rate warmup wrapper"""
    def __init__(self, optimizer, warmup_epochs, warmup_multiplier, after_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_multiplier = warmup_multiplier
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None, metric=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.warmup_multiplier + (1.0 - self.warmup_multiplier) * epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Use the main scheduler
            if isinstance(self.after_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.after_scheduler.step(metric)
            else:
                self.after_scheduler.step()
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def create_model(config_dict, device):
    """Create model from configuration"""
    model = SwinSR(**config_dict)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return model


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """Save checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)
        print(f"✓ Best model saved to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
    return start_epoch, best_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, writer, config):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    end = time.time()
    
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
        data_time.update(time.time() - end)
        
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.use_amp:
            with autocast('cuda', enabled=config.use_amp):
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            
            # Gradient clipping
            if config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            
            optimizer.step()
        
        losses.update(loss.item(), lr_imgs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Logging
        if batch_idx % config.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', losses.avg, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
    
    return losses.avg


def validate(model, dataloader, criterion, device, epoch, writer, config):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            
            if config.use_amp:
                with autocast('cuda', enabled=config.use_amp):
                    sr_imgs = model(lr_imgs)
                    loss = criterion(sr_imgs, hr_imgs)
            else:
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
            
            losses.update(loss.item(), lr_imgs.size(0))
            
            pbar.set_postfix({'val_loss': f'{losses.avg:.6f}'})
    
    writer.add_scalar('Validation/Loss', losses.avg, epoch)
    
    return losses.avg


def train(config):
    """Main training function"""
    # Display configuration
    display_config(config)
    
    # Set random seed
    set_seed(config.seed, config.deterministic)
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = TurbulenceSRDataset(
        root=config.data_root,
        upscale=config.upscale_factor,
        split='train',
        normalize=config.normalize_data
    )
    
    val_dataset = TurbulenceSRDataset(
        root=config.data_root,
        upscale=config.upscale_factor,
        split='test',  # Using test split as validation
        normalize=config.normalize_data
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config.model, device)
    
    # Create loss function
    criterion = get_loss_function(config.loss_type)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    base_scheduler = get_scheduler(optimizer, config.scheduler_type, config)
    
    # Wrap with warmup if needed
    if config.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            warmup_multiplier=config.warmup_multiplier,
            after_scheduler=base_scheduler
        )
    else:
        scheduler = base_scheduler
    
    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Resume from checkpoint if needed
    start_epoch = 0
    best_loss = float('inf')
    
    if config.resume and config.resume_checkpoint:
        if os.path.exists(config.resume_checkpoint):
            start_epoch, best_loss = load_checkpoint(
                config.resume_checkpoint, model, optimizer, scheduler
            )
        else:
            print(f"Warning: Checkpoint {config.resume_checkpoint} not found. Starting from scratch.")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, writer, config
        )
        
        # Validate
        val_loss = None
        if (epoch + 1) % config.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device, epoch, writer, config)
            print(f"\nEpoch {epoch}: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"\nEpoch {epoch}: Train Loss: {train_loss:.6f}")
        
        # Update learning rate
        if isinstance(scheduler, WarmupScheduler):
            scheduler.step(epoch, metric=val_loss)
        elif isinstance(base_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Save checkpoint
        is_best = False
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            is_best = True
        
        # Save regular checkpoint
        if (epoch + 1) % config.save_interval == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.after_scheduler.state_dict() if isinstance(scheduler, WarmupScheduler) else scheduler.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.to_dict()
            }
            
            if not config.save_best_only:
                save_checkpoint(
                    checkpoint_state,
                    is_best=False,
                    checkpoint_dir=config.checkpoint_dir,
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )
            
            if is_best:
                save_checkpoint(
                    checkpoint_state,
                    is_best=True,
                    checkpoint_dir=config.checkpoint_dir
                )
        
        print("-" * 80)
    
    writer.close()
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Model saved to: {config.checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swin Transformer Super-Resolution')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file (default: config.yaml)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu), overrides config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size, overrides config')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs, overrides config')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate, overrides config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.resume is not None:
        config.resume = True
        config.resume_checkpoint = args.resume
    
    # Start training
    train(config)


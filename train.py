import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, default_collate
import torchvision.transforms.v2 as transforms
import timm
import numpy as np
from tqdm import tqdm
import wandb

from accelerate import Accelerator
from datasets import load_dataset

from dataset import inDataset
from attnmixer import AttnMixer


torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(
        self,
        epochs=50,
        batch_size=128,
        lr=1e-4,
        save_dir='./checkpoints',
        wandb_logging=False,
        log_every=50,
        wandb_name='IN1k-Training', 
        checkpoint_dir=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir
        self.log_every = log_every

        n_classes = 1000
        
        self.accelerator = Accelerator(
            mixed_precision='bf16',
            dynamo_backend='INDUCTOR',
        )

        self.wandb_logging = wandb_logging
        if wandb_logging:
            wandb.init(project="IN1k-training", name=wandb_name)

        # Define transforms for training
        self.train_transform, self.test_transform = self.build_transforms()
        
        cutmix = transforms.CutMix(num_classes=n_classes)
        # mixup = transforms.MixUp(num_classes=n_classes)
        cutmix_or_mixup = transforms.RandomChoice([
            transforms.RandomApply([cutmix], p=0.2), 
            # transforms.RandomApply([mixup], p=0.2)
        ])
        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        self.train_dataset = inDataset(
            dataset='train',
            train_transforms=self.train_transform,
        )
        self.val_dataset = inDataset(
            dataset='validation',
            val_transforms=self.test_transform,
        )

        # Create DataLoader for training and validation
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            # num_workers=12,
            # pin_memory=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            # num_workers=12,
            # pin_memory=True,
        )
        
        self.model = AttnMixer(
            patch_embed_dim=128,
            num_heads=(4,4,5,5)
        )

        # self.model = torch.compile(self.model)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Set device and move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Use CrossEntropyLoss for single-label classification with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_test_loss = float('inf')
        self.best_test_acc = 0.0

        # Set up learning rate scheduler: warmup followed by cosine annealing
        steps_per_epoch = len(self.train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = 3 * steps_per_epoch
        decay_steps = total_steps - warmup_steps
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=decay_steps, eta_min=0.01 * self.lr
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
        )

        # Load prev training state if checkpoint_dir is provided
        if checkpoint_dir:
            self.accelerator.load_state(checkpoint_dir)
            print(f"Loaded training state from {checkpoint_dir}")

    def build_transforms(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.2, value='random', scale=(0.01, 0.2)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),
            ], p=0.15),      
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Define transforms for validation and testing (no augmentation)
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return train_transform, test_transform

    def save_model_if_best(self, val_loss, val_acc=0.0):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            os.makedirs(self.save_dir, exist_ok=True)

            # save train state
            self.accelerator.save_state(
                os.path.join(self.save_dir)
            )

    def run_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} - Training", leave=False)
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with self.accelerator.autocast():
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

            if i % self.log_every == 0:
                if self.wandb_logging:
                    wandb.log({
                        'train_loss': loss.item(),
                        'lr': self.scheduler.get_last_lr()[0],
                    })

        avg_loss = running_loss / total
        return avg_loss

    def run_validation(self):
        self.model.eval()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.inference_mode():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                # Top-1 accuracy
                _, predicted_top1 = torch.max(outputs, 1)
                correct_top1 += (predicted_top1 == labels).sum().item()

                # Top-5 accuracy
                top5_preds = torch.topk(outputs, k=5, dim=1).indices
                correct_top5 += sum([labels[i] in top5_preds[i] for i in range(labels.size(0))])

                total += labels.size(0)
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = running_loss / total
        top1_acc = correct_top1 / total
        top5_acc = correct_top5 / total

        return avg_loss, top1_acc, top5_acc

    def run_training(self):
        for epoch in tqdm(range(self.epochs), desc="Epochs", unit="epoch"):
            train_loss = self.run_epoch(epoch)
            print(f"Training - Epoch {epoch + 1}, Loss: {train_loss:.4f}")
            
            val_loss, val_acc, val_acc_top5 = self.run_validation()
            print(f"Validation - Top1 Acc: {val_acc:.4f}, Top5 Acc: {val_acc_top5:.4f}, Loss: {val_loss:.4f}")
            self.save_model_if_best(val_loss, val_acc)

            if self.wandb_logging:
                wandb.log({
                    # 'train_loss': train_loss, 
                    'val_loss': val_loss, 
                    'val_acc': val_acc, 
                    'val_acc_top5': val_acc_top5, 
                    'lr': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                })

        print(f"Training complete. Best validation metrics: Loss: {self.best_val_loss:.4f}, Top1 Acc: {self.best_val_acc:.4f}, Top5 Acc: {self.best_val_acc_top5:.4f}")


# Example usage:
if __name__ == '__main__':
    trainer = Trainer(
        epochs=300,
        batch_size=608,
        lr=0.002, # ---> 0.000005
        save_dir='./attn_mixer-b608-lr0.002',
        wandb_logging=False,  # set to True if you wish to log with wandb
        wandb_name='attn_mixer-b608-lr0.002', # wandb name
    )
    trainer.run_training()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import mlflow

from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import json

from src.logging import logger
from src.utils import count_parameters, create_paths
from src.config.manager import ConfigurationManager
from src.constants import DEVICE
from src.integrations.mlflow import MLFlowIntegration


class EarlyStopping:
    def __init__(
            self,
            patience: int=7,
            min_delta: float=0.0001
        ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ChessTrainerMLflow:
    def __init__(
        self,
        config: ConfigurationManager,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str=DEVICE
    ):
        self.config = config.get_model_trainer_config()
        
        self.model = model.to(device)
        self.model = torch.compile(self.model)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = GradScaler()

        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss(beta=self.config.beta)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-4
        )
        
        create_paths(self.config.checkpoint_dir)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        try:
            logger.info("Succesfuly connected to the MLFlow server: " + MLFlowIntegration.connect())
        except Exception as e:
            logger.exception("Unable to connect to the MLFlow server: " + str(e))
            raise e
        
    def calculate_metrics(self, predictions, targets):
        """Calculate additional metrics beyond loss."""
        predictions = predictions.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        mae = np.mean(np.abs(predictions - targets))
        
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        accuracy_01 = np.mean(np.abs(predictions - targets) < 0.1)
        accuracy_02 = np.mean(np.abs(predictions - targets) < 0.2)
        
        return {
            'mae': mae,
            'correlation': correlation,
            'accuracy_0.1': accuracy_01,
            'accuracy_0.2': accuracy_02
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (positions, evaluations) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            positions = positions.to(self.device, non_blocking=self.device == "cuda").to(torch.float32)
            evaluations = evaluations.to(self.device, non_blocking=self.device == "cuda")
            
            with autocast(device_type="cuda"):
                predictions = self.model(positions).squeeze(-1)
                loss = self.criterion(predictions, evaluations)
            
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(evaluations)
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            
            if batch_idx % 100 == 0:
                mlflow.log_metric('train_batch_loss', loss.item(), 
                                 step=epoch * len(self.train_loader) + batch_idx)
        
        
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for positions, evaluations in pbar:
                positions = positions.to(self.device, non_blocking=self.device == "cuda").to(torch.float32)
                evaluations = evaluations.to(self.device, non_blocking=self.device == "cuda")
                
                predictions = self.model(positions)
                loss = self.criterion(predictions, evaluations)
                
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(evaluations)
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best model (val_loss: {val_loss:.6f})')
            
            try:
                mlflow.pytorch.log_model(self.model, name="best_model")
            except Exception as e:
                logger.exception("Failed to log the model: " + str(e))
                #raise e
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def train(self):
        """Full training loop with MLflow tracking."""
        
        
        with mlflow.start_run(run_name=self.model_name) as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            
            mlflow.log_params({
                'model_type': 'ResNet',
                'num_epochs': self.config.num_epochs,
                'batch_size': self.train_loader.batch_size,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                'optimizer': 'AdamW',
                'loss': 'SmootL1Loss',
                'beta': self.config.beta,
                'scheduler': 'CosineAnnealingLR',
                'early_stopping_patience': self.config.early_stopping_patience,
                'device': str(self.device),
                'num_parameters': count_parameters(self.model),
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset)
            })
            
            
            if hasattr(self.model, 'num_blocks'):
                mlflow.log_param('num_blocks', self.model.num_blocks)
            if hasattr(self.model, 'num_filters'):
                mlflow.log_param('num_filters', self.model.num_filters)
            
            
            model_summary = str(self.model)
            with open(self.checkpoint_dir / 'model_summary.txt', 'w') as f:
                f.write(model_summary)
            mlflow.log_artifact(str(self.checkpoint_dir / 'model_summary.txt'))
            
            logger.info(f"\nStarting training for {self.config.num_epochs} epochs...")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model parameters: {count_parameters(self.model):,}")
            logger.info(f"Training batches: {len(self.train_loader)}")
            logger.info(f"Validation batches: {len(self.val_loader)}")
            
            early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
            best_val_loss = float('inf')
            
            
            self.scheduler.T_max = self.config.num_epochs
            
            start_time = time.time()
            
            for epoch in range(1, self.config.num_epochs + 1):
                epoch_start = time.time()
                
                
                train_loss, train_metrics = self.train_epoch(epoch)
                
                
                val_loss, val_metrics = self.validate(epoch)
                
                
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rate'].append(current_lr)
                
                
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_mae': train_metrics['mae'],
                    'train_correlation': train_metrics['correlation'],
                    'train_accuracy_0.1': train_metrics['accuracy_0.1'],
                    'train_accuracy_0.2': train_metrics['accuracy_0.2'],
                    'val_loss': val_loss,
                    'val_mae': val_metrics['mae'],
                    'val_correlation': val_metrics['correlation'],
                    'val_accuracy_0.1': val_metrics['accuracy_0.1'],
                    'val_accuracy_0.2': val_metrics['accuracy_0.2'],
                    'learning_rate': current_lr,
                }, step=epoch)
                
                epoch_time = time.time() - epoch_start
                logger.info(f"\nEpoch {epoch}/{self.config.num_epochs} - {epoch_time:.1f}s")
                logger.info(f"  Train Loss: {train_loss:.6f} | MAE: {train_metrics['mae']:.6f} | "
                      f"Corr: {train_metrics['correlation']:.4f}")
                logger.info(f"  Val Loss:   {val_loss:.6f} | MAE: {val_metrics['mae']:.6f} | "
                      f"Corr: {val_metrics['correlation']:.4f}")
                logger.info(f"  LR:         {current_lr:.2e}")
                
                
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    
                    mlflow.log_metrics({
                        'best_val_loss': best_val_loss,
                        'best_val_correlation': val_metrics['correlation'],
                        'best_epoch': epoch
                    })
                
                self.save_checkpoint(epoch, val_loss, is_best)
                
                
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    mlflow.log_metric('early_stopped_epoch', epoch)
                    break
                
            
            total_time = time.time() - start_time
            logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")
            logger.info(f"Best validation loss: {best_val_loss:.6f}")
            
            
            mlflow.log_metric('total_training_time_minutes', total_time / 60)
            mlflow.log_metric('final_val_loss', val_loss)
            
            logger.info("Saving training history...")
            history_path = self.checkpoint_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            mlflow.log_artifact(str(history_path))
            
            logger.info("Saving checkpoints...")
            mlflow.log_artifacts(str(self.checkpoint_dir), artifact_path="checkpoints")
            
            logger.info("Generating loss curve...")
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(self.history['train_loss'], label='Train Loss', alpha=0.7)
                plt.plot(self.history['val_loss'], label='Val Loss', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_path = self.checkpoint_dir / 'loss_curve.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(str(plot_path))
                logger.info(f"Loss curve saved and logged to MLflow")
            except ImportError:
                logger.exception("Matplotlib not available, skipping loss curve plot")
import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import OrderedDict
import heapq
import hashlib
from pathlib import Path
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelCheckpoint:
    """Class to store model checkpoint information"""
    epoch: int
    validation_loss: float
    training_loss: float
    combined_loss: float
    save_path: str
    hyperparameters: Dict

    def to_dict(self) -> Dict:
        """Convert checkpoint to dictionary for serialization"""
        return {
            'epoch': self.epoch,
            'validation_loss': self.validation_loss,
            'training_loss': self.training_loss,
            'combined_loss': self.combined_loss,
            'save_path': self.save_path,
            'hyperparameters': self.hyperparameters
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelCheckpoint':
        """Create checkpoint from dictionary"""
        return cls(**data)

class ModelManager:
    """Model manager for saving and loading best models with persistent tracking"""
    
    def __init__(self, 
                 base_dir: str,
                 max_checkpoints_per_metric: int = 2,
                 hyperparameters: Dict = None,
                 train_env_name: str = None,
                 seed: int = None):
        """
        Initialize the model manager
        
        Args:
            base_dir: Base directory for model storage
            max_checkpoints_per_metric: Number of best models to keep per metric
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints_per_metric
        self.best_models: Dict[str, Dict] = {}  # config_hash -> metric_type -> checkpoints
        self.hyperparameters = hyperparameters
        #self.tracking_file = self.base_dir / "model_tracking.json"
        self.model_dir = self._get_model_dir(train_env_name, seed, self.hyperparameters)
        self.tracking_file = self.model_dir / "model_tracking.json"
        
        # Load existing tracking data
        self._load_tracking_data()
    
    @staticmethod
    def get_config_hash(train_env_name: str, seed: int, hyperparameters: Dict) -> str:
        """Generate hash for internal tracking"""
        config_str = (f"{train_env_name}_{seed}_" + 
                     '_'.join(f"{k}_{v}" for k, v in sorted(hyperparameters.items())))
        return hashlib.md5(config_str.encode()).hexdigest()[:10]
    
    @staticmethod
    def get_readable_folder_name(train_env_name: str, seed: int, hyperparameters: Dict) -> str:
        """Generate readable folder name from configuration"""
        # Start with environment and seed
        folder_name = f"{train_env_name}_seed{seed}"
        
        # Add important hyperparameters
        param_strs = []
        for k, v in sorted(hyperparameters.items()):
            if isinstance(v, float):
                # Format floating point numbers nicely
                param_str = f"{k}{v:.0e}" if v < 0.01 else f"{k}{v}"
            else:
                param_str = f"{k}{v}"
            param_strs.append(param_str)
        
        if param_strs:
            folder_name += "_" + "_".join(param_strs)
        
        # Clean up any characters that might cause issues
        folder_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in folder_name)
        return folder_name
    
    def _get_model_dir(self, train_env_name: str, seed: int, hyperparameters: Dict) -> Path:
        """Get directory for specific configuration"""
        folder_name = self.get_readable_folder_name(train_env_name, seed,self.hyperparameters)
        model_dir = self.base_dir / folder_name
        model_dir.mkdir(exist_ok=True)
        
        # Save hyperparameter info
        self._save_config_info(model_dir, train_env_name, seed)#, self.hyperparameters)
        return model_dir
    
    def _save_config_info(self, model_dir: Path, train_env_name: str, 
                         seed: int):#, hyperparameters: Dict):
        """Save configuration information in the model directory"""
        info_file = model_dir / "config_info.json"
        info = {
            "train_env_name": train_env_name,
            "seed": seed,
            "hyperparameters": self.hyperparameters,
            "config_hash": self.get_config_hash(train_env_name, seed, self.hyperparameters),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_tracking_data(self):
        """Load best models tracking data from disk"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    tracking_data = json.load(f)
                
                # Reconstruct best_models dictionary
                for config_hash, metrics in tracking_data.items():
                    self.best_models[config_hash] = {
                        'validation': [],
                        'training': [],
                        'combined': []
                    }
                    
                    for metric, checkpoints in metrics.items():
                        for ckpt_data in checkpoints:
                            checkpoint = ModelCheckpoint.from_dict(ckpt_data['checkpoint'])
                            loss = ckpt_data['loss']
                            heapq.heappush(self.best_models[config_hash][metric],
                                         (loss, checkpoint))
                
                logger.info(f"Loaded tracking data for {len(self.best_models)} configurations")
            except Exception as e:
                logger.error(f"Error loading tracking data: {e}")
                self.best_models = {}
    
    def _save_tracking_data(self):
        """Save best models tracking data to disk"""
        try:
            tracking_data = {}
            for config_hash, metrics in self.best_models.items():
                tracking_data[config_hash] = {
                    'validation': [{'loss': loss, 'checkpoint': ckpt.to_dict()} 
                                 for loss, ckpt in sorted(metrics['validation'])],
                    'training': [{'loss': loss, 'checkpoint': ckpt.to_dict()} 
                               for loss, ckpt in sorted(metrics['training'])],
                    'combined': [{'loss': loss, 'checkpoint': ckpt.to_dict()} 
                               for loss, ckpt in sorted(metrics['combined'])]
                }
            
            with open(self.tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            
            logger.debug("Saved tracking data successfully")
        except Exception as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def _initialize_best_models(self, config_hash: str):
        """Initialize tracking for a new configuration"""
        if config_hash not in self.best_models:
            self.best_models[config_hash] = {
                'validation': [],
                'training': [],
                'combined': []
            }
    
    def _update_best_models(self, checkpoint: ModelCheckpoint, 
                          config_hash: str) -> List[str]:
        """Update best models tracking"""
        metrics_updated = []
        
        # Update validation loss models
        if self._update_metric_models(checkpoint, 
                                    self.best_models[config_hash]['validation'],
                                    lambda x: -x.validation_loss):
            metrics_updated.append('validation')
        
        # Update training loss models
        if self._update_metric_models(checkpoint, 
                                    self.best_models[config_hash]['training'],
                                    lambda x: -x.training_loss):
            metrics_updated.append('training')
        
        # Update combined loss models
        if self._update_metric_models(checkpoint, 
                                    self.best_models[config_hash]['combined'],
                                    lambda x: -x.combined_loss):
            metrics_updated.append('combined')
        
        return metrics_updated
    
    def _update_metric_models(self, checkpoint: ModelCheckpoint, 
                            model_list: List, 
                            key_func) -> bool:
        """Update models for a specific metric"""
        key_value = key_func(checkpoint)
        
        # Check if this would be among the best
        if len(model_list) >= self.max_checkpoints:
            worst_key = model_list[0][0]
            if key_value <= worst_key:
                return False
        
        # Add new checkpoint
        heapq.heappush(model_list, (key_value, checkpoint))
        
        # Keep only the best ones
        while len(model_list) > self.max_checkpoints:
            heapq.heappop(model_list)
        
        return True
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       train_env_name: str,
                       seed: int,
                       #hyperparameters: Dict,
                       losses: Dict[str, float]) -> str:
        """Save model checkpoint if it's among the best"""
        # Get config hash and model directory
        config_hash = self.get_config_hash(train_env_name, seed, self.hyperparameters)
        
        # Initialize tracking if needed
        self._initialize_best_models(config_hash)
        
        # Generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(self.model_dir / f"model_e{epoch}_{timestamp}.pt")
        
        # Create checkpoint object
        checkpoint = ModelCheckpoint(
            epoch=epoch,
            validation_loss=losses['val'],
            training_loss=losses['train'],
            combined_loss=losses['val'] + losses['train'],
            save_path=save_path,
            hyperparameters=self.hyperparameters
        )
        
        # Update best models tracking
        metrics_updated = self._update_best_models(checkpoint, config_hash)
        
        if metrics_updated:
            # Save model state
            state_save = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'hyperparameters':self.hyperparameters,
                'losses': losses
            }
            torch.save(state_save, save_path)
            
            # Save tracking data
            self._save_tracking_data()
            
            logger.info(f"Saved new best model for metrics: {', '.join(metrics_updated)}")
            logger.info(f"Model saved to: {save_path}")
        
        return save_path
    
    def get_best_model_info(self,
                           train_env_name: str,
                           seed: int,
                           hyperparameters: Dict) -> Dict[str, List[Dict]]:
        """Get information about best models for a configuration"""
        if not self.tracking_file.exists():
            logger.warning("No tracking file found")
            return {}
        
        try:
            with open(self.tracking_file, 'r') as f:
                tracking_data = json.load(f)
            
            config_hash = self.get_config_hash(train_env_name, seed, hyperparameters)
            
            if config_hash not in tracking_data:
                logger.warning(f"No models found for configuration {config_hash}")
                return {}
            
            info = {}
            for metric in ['validation', 'training', 'combined']:
                info[metric] = [
                    {
                        'epoch': ckpt['checkpoint']['epoch'],
                        'validation_loss': ckpt['checkpoint']['validation_loss'],
                        'training_loss': ckpt['checkpoint']['training_loss'],
                        'combined_loss': ckpt['checkpoint']['combined_loss'],
                        'save_path': ckpt['checkpoint']['save_path']
                    }
                    for ckpt in sorted(tracking_data[config_hash][metric],
                                     key=lambda x: x['loss'])
                ]
            
            return info
        except Exception as e:
            logger.error(f"Error reading tracking data: {e}")
            return {}
    
    def load_best_models(self,
                        train_env_name: str,
                        seed: int,
                        hyperparameters: Dict,
                        metric: str = 'validation',
                        device: str = "cuda:0") -> List[Dict]:
        """Load best models for a configuration"""
        model_info = self.get_best_model_info(train_env_name, seed, hyperparameters)
        
        if not model_info or metric not in model_info:
            return []
        
        results = []
        for model in model_info[metric]:
            try:
                if not os.path.exists(model['save_path']):
                    logger.warning(f"Model file not found: {model['save_path']}")
                    continue
                    
                state_dict = torch.load(model['save_path'], map_location=device)['state_dict']
                results.append({
                    'epoch': model['epoch'],
                    'state_dict': state_dict,
                    'validation_loss': model['validation_loss'],
                    'training_loss': model['training_loss'],
                    'combined_loss': model['combined_loss'],
                    'save_path': model['save_path']
                })
                
                logger.info(f"Successfully loaded model from epoch {model['epoch']}")
            except Exception as e:
                logger.error(f"Error loading model from {model['save_path']}: {e}")
                continue
        
        return results
    
    def get_all_saved_configs(self) -> List[Dict]:
        """Get information about all saved configurations"""
        try:
            if not self.tracking_file.exists():
                logger.warning("No tracking file found")
                return []
                
            with open(self.tracking_file, 'r') as f:
                tracking_data = json.load(f)
            
            configs = []
            for config_hash, metrics in tracking_data.items():
                sample_checkpoint = metrics['validation'][0]['checkpoint']
                configs.append({
                    'config_hash': config_hash,
                    'hyperparameters': sample_checkpoint['hyperparameters'],
                    'num_models': {
                        metric: len(checkpoints) 
                        for metric, checkpoints in metrics.items()
                    }
                })
            
            return configs
        except Exception as e:
            logger.error(f"Error reading tracking data: {e}")
            return []
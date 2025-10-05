"""
Entrenador de modelo de Deep Learning para clasificación PERRO vs NO-PERRO
Optimizado para GPU AMD 7900XTX con ROCm
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.models as models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DogClassificationModel(nn.Module):
    """Modelo de clasificación binaria basado en arquitecturas preentrenadas"""
    
    def __init__(self, model_name: str = 'efficientnet_b3', num_classes: int = 1, pretrained: bool = True):
        super(DogClassificationModel, self).__init__()
        
        self.model_name = model_name
        
        # Seleccionar arquitectura base
        if model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
            
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        # Cabezal clasificador personalizado
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Inicializar pesos del clasificador
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Inicializa los pesos del clasificador"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()
    
    def freeze_backbone(self, freeze: bool = True):
        """Congela/descongela el backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

class ModelTrainer:
    """Entrenador del modelo con métricas avanzadas"""
    
    def __init__(self, model_name: str = 'efficientnet_b3', device: str = 'auto'):
        # Configurar dispositivo (ROCm para AMD)
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
                print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.device = torch.device('cpu')
                print("⚠️  Usando CPU (no se detectó GPU compatible)")
        else:
            self.device = torch.device(device)
        
        # Crear modelo
        self.model = DogClassificationModel(model_name=model_name)
        self.model.to(self.device)
        
        # Configurar para entrenamiento mixto de precisión
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Métricas de entrenamiento
        self.train_history = {
            'loss': [], 'accuracy': [], 'lr': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
    def setup_training(self, train_loader, val_loader, class_weights=None):
        """Configura optimizador, loss y scheduler"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Calculer class weights si no se proporcionan
        if class_weights is None:
            pos_weight = self._calculate_pos_weight(train_loader)
        else:
            pos_weight = torch.tensor(class_weights[1] / class_weights[0])
        
        pos_weight = pos_weight.to(self.device)
        
        # Loss function con class weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizador AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        print(f"✅ Configuración de entrenamiento completada")
        print(f"   Dispositivo: {self.device}")
        print(f"   Pos weight: {pos_weight.item():.3f}")
        print(f"   Parámetros totales: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Parámetros entrenables: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def _calculate_pos_weight(self, train_loader):
        """Calcula el peso positivo para balancear clases"""
        total_negative = 0
        total_positive = 0
        
        for _, labels in train_loader:
            total_positive += labels.sum().item()
            total_negative += (1 - labels).sum().item()
        
        pos_weight = total_negative / max(total_positive, 1)
        return torch.tensor(pos_weight)
    
    def train_epoch(self, epoch: int):
        """Entrena una época"""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Época {epoch+1}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass con mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Métricas
            running_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Actualizar progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return epoch_loss, epoch_accuracy, current_lr
    
    def validate_epoch(self):
        """Valida el modelo"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validación'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Recopilar predicciones y probabilidades
                probabilities = torch.sigmoid(outputs)
                predictions = probabilities > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Métricas
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_auc = roc_auc_score(all_labels, all_probabilities)
        
        # Métricas adicionales
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        return val_loss, val_accuracy, val_auc, precision, recall, f1
    
    def train_model(self, num_epochs: int = 50, save_path: str = None, 
                   freeze_epochs: int = 5):
        """Entrena el modelo completo"""
        print(f"🚀 Iniciando entrenamiento por {num_epochs} épocas...")
        print(f"   Primeras {freeze_epochs} épocas: backbone congelado")
        print("="*60)
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            self.best_model_path = save_path / 'best_model.pth'
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Descongelar backbone después de freeze_epochs
            if epoch == freeze_epochs:
                print(f"\n🔓 Descongelando backbone en época {epoch+1}")
                self.model.freeze_backbone(False)
                # Reducir learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            
            # Entrenar época
            train_loss, train_acc, current_lr = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = self.validate_epoch()
            
            # Actualizar scheduler
            self.scheduler.step(val_acc)
            
            # Guardar métricas
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['lr'].append(current_lr)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['val_auc'].append(val_auc)
            
            # Guardar mejor modelo
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                if save_path:
                    self.save_model(self.best_model_path, epoch, val_acc)
            
            # Imprimir métricas
            print(f"\nÉpoca {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            print(f"  Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Early stopping simple
            if current_lr < 1e-6:
                print("⏹️  Early stopping: Learning rate muy bajo")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Entrenamiento completado en {elapsed_time/60:.1f} minutos")
        print(f"   Mejor validación accuracy: {self.best_val_accuracy:.4f}")
        
        if save_path:
            self.save_training_history(save_path)
            self.plot_training_curves(save_path)
        
        return self.train_history
    
    def save_model(self, save_path: Path, epoch: int, val_accuracy: float):
        """Guarda el modelo"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'model_name': self.model.model_name,
            'train_history': self.train_history
        }, save_path)
        
        print(f"💾 Modelo guardado: {save_path} (val_acc: {val_accuracy:.4f})")
    
    def load_model(self, model_path: str):
        """Carga un modelo guardado"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        
        print(f"📁 Modelo cargado desde: {model_path}")
        print(f"   Época: {checkpoint['epoch']}, Val Acc: {checkpoint['val_accuracy']:.4f}")
    
    def save_training_history(self, save_path: Path):
        """Guarda el historial de entrenamiento"""
        history_path = save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"📊 Historial guardado: {history_path}")
    
    def plot_training_curves(self, save_path: Path):
        """Crea gráficos de las curvas de entrenamiento"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Curvas de Entrenamiento', fontsize=16)
        
        epochs = range(1, len(self.train_history['loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.train_history['loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.train_history['accuracy'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.train_history['val_accuracy'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.train_history['lr'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # AUC
        axes[1, 1].plot(epochs, self.train_history['val_auc'], 'purple')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = save_path / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Gráficos guardados: {plot_path}")

def setup_rocm_optimization():
    """Configura optimizaciones específicas para ROCm/AMD"""
    # Configuraciones para ROCm
    if torch.cuda.is_available():
        print("🔧 Configurando optimizaciones ROCm...")
        
        # Configurar memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Configurar para mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ Optimizaciones ROCm configuradas")
        
        return True
    else:
        print("⚠️  ROCm no disponible, usando configuración CPU")
        return False

if __name__ == "__main__":
    # Configurar ROCm
    rocm_available = setup_rocm_optimization()
    
    # Configurar modelo
    model_name = 'efficientnet_b3'  # Opciones: 'efficientnet_b3', 'resnet50', 'resnet101', 'densenet121'
    
    print(f"🤖 Configurando modelo: {model_name}")
    print("="*60)
    
    # Crear trainer
    trainer = ModelTrainer(model_name=model_name)
    
    print(f"\n🏗️  Modelo creado:")
    print(f"   Arquitectura: {model_name}")
    print(f"   Dispositivo: {trainer.device}")
    print(f"   Mixed Precision: {'Sí' if trainer.scaler else 'No'}")
    
    # Ejemplo de uso con DataLoaders (necesita ejecutar data_preprocessor.py primero)
    # from data_preprocessor import DataPreprocessor
    # 
    # preprocessor = DataPreprocessor(dataset_path, output_path)
    # data_loaders, splits = preprocessor.process_complete_dataset()
    # 
    # trainer.setup_training(data_loaders['train'], data_loaders['val'])
    # history = trainer.train_model(num_epochs=30, save_path='./models')
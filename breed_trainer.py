"""
Trainer Especializado para Clasificación de 50 Razas de Perros
Optimizado para AMD 7800X3D con técnicas avanzadas
"""

import os
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import pandas as pd
from tqdm import tqdm

class AdvancedBreedClassifier(nn.Module):
    """Modelo avanzado para clasificación de razas con técnicas modernas"""
    
    def __init__(self, num_classes=50, model_name='efficientnet_b3', pretrained=True):
        super(AdvancedBreedClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Crear backbone
        if model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'convnext_small':
            self.backbone = models.convnext_small(pretrained=pretrained)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        # Classifier Head avanzado con regularización
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicialización de pesos optimizada"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extrae características para análisis"""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        return features

class BreedTrainer:
    """Trainer avanzado para clasificación de razas"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=50, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Crear modelo
        self.model = AdvancedBreedClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=True
        ).to(device)
        
        # Configuración de entrenamiento
        self.best_val_acc = 0.0
        self.best_val_top5 = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'train_top5': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5': [],
            'learning_rates': []
        }
        
        # Cargar configuración de razas
        try:
            from breed_processed_data.dataset_config import DATASET_INFO, INDEX_TO_DISPLAY
            self.breed_names = INDEX_TO_DISPLAY
            self.dataset_info = DATASET_INFO
            print("✅ Configuración de razas cargada")
        except ImportError:
            print("⚠️  Configuración de razas no encontrada")
            self.breed_names = {i: f"Breed_{i}" for i in range(num_classes)}
            self.dataset_info = {}
        
        # Configuraciones optimizadas para 7800X3D
        self.setup_environment()
    
    def setup_environment(self):
        """Configura el entorno para 7800X3D"""
        # Variables de entorno ya configuradas por el preprocessor
        
        # Configuraciones de PyTorch para CPU
        torch.set_num_threads(16)
        torch.set_num_interop_threads(16)
        
        # Habilitar optimizaciones
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        print("🚀 Entorno optimizado para 7800X3D")
    
    def setup_training(self, train_loader, val_loader, learning_rate=1e-3, weight_decay=1e-4):
        """Configura optimizador y scheduler"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizador AdamW con configuración optimizada
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * 40  # 40 épocas estimadas
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 5,  # Peak LR
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            div_factor=25,  # Initial LR = max_lr/25
            final_div_factor=10000  # Final LR = max_lr/10000
        )
        
        # Función de pérdida con label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Scaler para mixed precision (aunque estemos en CPU)
        self.scaler = GradScaler(enabled=False)  # CPU no soporta AMP
        
        print("⚙️  Configuración de entrenamiento lista")
        print(f"   📈 Learning rate: {learning_rate}")
        print(f"   📊 Total steps: {total_steps:,}")
        print(f"   🎯 Clases: {self.num_classes}")
    
    def calculate_metrics(self, outputs, targets):
        """Calcula métricas de evaluación"""
        predictions = torch.argmax(outputs, dim=1)
        
        # Accuracy top-1
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        
        # Accuracy top-5
        _, top5_pred = torch.topk(outputs, min(5, outputs.size(1)), dim=1)
        top5_correct = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
        top5_accuracy = top5_correct.any(dim=1).float().mean().item()
        
        return accuracy, top5_accuracy
    
    def train_epoch(self, epoch):
        """Entrena una época"""
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        running_top5 = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch+1}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=False):  # CPU no soporta autocast
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Métricas
            acc, top5_acc = self.calculate_metrics(outputs, targets)
            
            # Actualizar estadísticas
            running_loss += loss.item()
            running_acc += acc
            running_top5 += top5_acc
            
            # Actualizar progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.3f}',
                'Top5': f'{top5_acc:.3f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Promediar métricas
        avg_loss = running_loss / len(self.train_loader)
        avg_acc = running_acc / len(self.train_loader)
        avg_top5 = running_top5 / len(self.train_loader)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Guardar en historial
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['train_acc'].append(avg_acc)
        self.training_history['train_top5'].append(avg_top5)
        self.training_history['learning_rates'].append(current_lr)
        
        return avg_loss, avg_acc, avg_top5, current_lr
    
    def validate_epoch(self):
        """Valida el modelo"""
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        running_top5 = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validación'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                acc, top5_acc = self.calculate_metrics(outputs, targets)
                
                running_loss += loss.item()
                running_acc += acc
                running_top5 += top5_acc
                
                # Guardar para métricas detalladas
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Promediar métricas
        avg_loss = running_loss / len(self.val_loader)
        avg_acc = running_acc / len(self.val_loader)
        avg_top5 = running_top5 / len(self.val_loader)
        
        # Guardar en historial
        self.training_history['val_loss'].append(avg_loss)
        self.training_history['val_acc'].append(avg_acc)
        self.training_history['val_top5'].append(avg_top5)
        
        return avg_loss, avg_acc, avg_top5, all_predictions, all_targets
    
    def save_checkpoint(self, epoch, save_path, is_best=False):
        """Guarda checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'training_history': self.training_history,
            'model_config': {
                'num_classes': self.num_classes,
                'model_name': self.model_name,
                'breed_names': self.breed_names
            }
        }
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Guardar checkpoint regular
        checkpoint_path = save_path / f'breed_model_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = save_path / 'best_breed_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Mejor modelo guardado: {best_path}")
        
        return checkpoint_path
    
    def create_confusion_matrix(self, predictions, targets, save_path=None):
        """Crea matriz de confusión"""
        cm = confusion_matrix(targets, predictions)
        
        # Crear figura
        plt.figure(figsize=(15, 12))
        
        # Matriz de confusión normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Usar solo primeras 20 clases para visualización clara
        top_20_indices = np.argsort(np.diag(cm))[-20:]
        cm_subset = cm_normalized[np.ix_(top_20_indices, top_20_indices)]
        
        # Plot
        sns.heatmap(cm_subset, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=[self.breed_names.get(i, f'Class_{i}') for i in top_20_indices],
                   yticklabels=[self.breed_names.get(i, f'Class_{i}') for i in top_20_indices])
        
        plt.title('Matriz de Confusión - Top 20 Clases', fontsize=14, fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matriz de confusión guardada: {save_path}")
        
        plt.close()
    
    def plot_training_history(self, save_path=None):
        """Grafica historial de entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Entrenamiento')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validación')
        ax1.set_title('Pérdida por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy Top-1
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Entrenamiento')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validación')
        ax2.set_title('Accuracy Top-1 por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Accuracy Top-5
        ax3.plot(epochs, self.training_history['train_top5'], 'b-', label='Entrenamiento')
        ax3.plot(epochs, self.training_history['val_top5'], 'r-', label='Validación')
        ax3.set_title('Accuracy Top-5 por Época')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate
        ax4.plot(epochs, self.training_history['learning_rates'], 'g-')
        ax4.set_title('Learning Rate por Época')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Historial de entrenamiento guardado: {save_path}")
        
        plt.close()
    
    def train_model(self, num_epochs=30, save_path='./breed_models', patience=7):
        """Entrena el modelo completo"""
        start_time = time.time()
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        print("🚀 INICIANDO ENTRENAMIENTO DE RAZAS")
        print("="*60)
        print(f"🎯 Épocas: {num_epochs}")
        print(f"🏷️  Clases: {self.num_classes}")
        print(f"🤖 Modelo: {self.model_name}")
        print(f"💻 Dispositivo: {self.device}")
        
        # Contador para early stopping
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 ÉPOCA {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Entrenamiento
            train_loss, train_acc, train_top5, current_lr = self.train_epoch(epoch)
            
            # Validación
            val_loss, val_acc, val_top5, predictions, targets = self.validate_epoch()
            
            # Mostrar resultados
            print(f"📊 Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, Top5: {train_top5:.3f}")
            print(f"✅ Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Top5: {val_top5:.3f}")
            print(f"📈 Learning Rate: {current_lr:.2e}")
            
            # Verificar si es el mejor modelo
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_top5 = val_top5
                patience_counter = 0
                print(f"🏆 ¡Nuevo mejor modelo! Acc: {val_acc:.3f}, Top5: {val_top5:.3f}")
            else:
                patience_counter += 1
            
            # Guardar checkpoint
            self.save_checkpoint(epoch, save_path, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping activado después de {patience} épocas sin mejora")
                break
        
        # Entrenamiento completado
        elapsed_time = time.time() - start_time
        
        # Crear visualizaciones finales
        self.plot_training_history(save_path / 'training_history.png')
        self.create_confusion_matrix(predictions, targets, save_path / 'confusion_matrix.png')
        
        # Guardar configuración final
        final_config = {
            'training_completed': True,
            'total_epochs': epoch + 1,
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'training_time_hours': elapsed_time / 3600,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'breed_names': self.breed_names
        }
        
        with open(save_path / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎯 ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"⏱️  Tiempo total: {elapsed_time/3600:.2f} horas")
        print(f"🏆 Mejor Acc: {self.best_val_acc:.3f}")
        print(f"🎯 Mejor Top-5: {self.best_val_top5:.3f}")
        print(f"📁 Modelos guardados en: {save_path}")
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_val_top5': self.best_val_top5,
            'training_time': elapsed_time,
            'total_epochs': epoch + 1,
            'save_path': save_path
        }

def main():
    """Función principal para entrenar el modelo de razas"""
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Usando dispositivo: {device}")
    
    # Cargar datos procesados
    try:
        from breed_preprocessor import BreedDatasetPreprocessor
        
        yesdog_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS\YESDOG"
        preprocessor = BreedDatasetPreprocessor(yesdog_path)
        
        print("🔄 Cargando dataset procesado...")
        # Los datos ya están procesados, solo crear DataLoaders
        from breed_processed_data.dataset_config import DATASET_INFO
        
        # Crear DataLoaders usando el preprocessor
        results = preprocessor.run_complete_preprocessing(target_samples_per_class=200)
        data_loaders = results['data_loaders']
        
        print(f"✅ Dataset cargado:")
        print(f"   🏋️  Train: {len(data_loaders['train'].dataset)} samples")
        print(f"   ✅ Val: {len(data_loaders['val'].dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        print("👉 Ejecuta primero breed_preprocessor.py")
        return None
    
    # Crear trainer
    trainer = BreedTrainer(
        model_name='efficientnet_b3',
        num_classes=50,
        device=device
    )
    
    # Configurar entrenamiento
    trainer.setup_training(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Entrenar modelo
    results = trainer.train_model(
        num_epochs=30,
        save_path='./breed_models',
        patience=7
    )
    
    return results

if __name__ == "__main__":
    # Fijar semillas para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    results = main()
    
    if results:
        print(f"\n🎉 ¡Entrenamiento exitoso!")
        print(f"🏆 Mejor accuracy: {results['best_val_acc']:.3f}")
        print(f"⏱️  Tiempo total: {results['training_time']/3600:.2f} horas")
    else:
        print("❌ Error en el entrenamiento")
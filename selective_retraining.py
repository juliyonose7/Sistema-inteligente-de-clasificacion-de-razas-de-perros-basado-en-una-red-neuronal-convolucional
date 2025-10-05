#!/usr/bin/env python3
"""
Reentrenamiento selectivo de razas problemáticas
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import random
import shutil
from pathlib import Path

class SelectiveBreedDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_breeds=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_breeds = target_breeds or []
        
        self.samples = []
        self.class_to_idx = {}
        
        # Solo usar razas objetivo
        available_breeds = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) and d in target_breeds]
        
        for idx, breed in enumerate(available_breeds):
            self.class_to_idx[breed] = idx
            breed_path = os.path.join(data_dir, breed)
            
            for img_file in os.listdir(breed_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(breed_path, img_file), idx))
        
        print(f"📊 Dataset creado: {len(self.samples)} imágenes, {len(available_breeds)} clases")
        for breed, idx in self.class_to_idx.items():
            count = sum(1 for s in self.samples if s[1] == idx)
            print(f"   {idx}: {breed} - {count} imágenes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # Devolver imagen aleatoria válida
            return self.__getitem__((idx + 1) % len(self.samples))

class SelectiveBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def create_selective_fine_tuning():
    print("🔧 REENTRENAMIENTO SELECTIVO - RAZAS PROBLEMÁTICAS")
    print("=" * 70)
    
    # Razas problemáticas identificadas
    target_breeds = [
        'Labrador_retriever',
        'Norwegian_elkhound', 
        'beagle',
        'pug',
        'basset',
        'Samoyed'  # Agregar algunas más para balance
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo: {device}")
    
    # Transformaciones de entrenamiento con data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Crear datasets
    train_dataset = SelectiveBreedDataset(
        'breed_processed_data/train', 
        transform=train_transform,
        target_breeds=target_breeds
    )
    
    # Crear directorio de validación si no existe
    val_dir = 'breed_processed_data/val'
    if not os.path.exists(val_dir):
        print("📁 Creando dataset de validación...")
        os.makedirs(val_dir, exist_ok=True)
        
        # Mover 20% de imágenes a validación
        for breed in target_breeds:
            breed_train = f'breed_processed_data/train/{breed}'
            breed_val = f'{val_dir}/{breed}'
            
            if os.path.exists(breed_train):
                os.makedirs(breed_val, exist_ok=True)
                
                images = [f for f in os.listdir(breed_train) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Mover 20% aleatoriamente
                val_count = max(1, len(images) // 5)
                val_images = random.sample(images, val_count)
                
                for img in val_images:
                    src = os.path.join(breed_train, img)
                    dst = os.path.join(breed_val, img)
                    shutil.move(src, dst)
                
                print(f"   {breed}: {len(val_images)} → validación")
    
    val_dataset = SelectiveBreedDataset(
        val_dir,
        transform=val_transform,
        target_breeds=target_breeds
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Modelo
    num_classes = len(train_dataset.class_to_idx)
    model = SelectiveBreedClassifier(num_classes).to(device)
    
    # Cargar modelo preentrenado como punto de partida
    pretrained_path = "autonomous_breed_models/best_breed_model_epoch_17_acc_0.9199.pth"
    if os.path.exists(pretrained_path):
        print("🔄 Cargando modelo preentrenado como punto de partida...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Solo cargar pesos del backbone (sin la capa final)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filtrar solo pesos del backbone
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'fc' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("✅ Backbone cargado, reentrenando clasificador")
    
    # Configuración de entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Entrenamiento
    print(f"\n🚀 Iniciando fine-tuning ({num_classes} clases)...")
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(20):  # Menos épocas para fine-tuning
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/20, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/20:")
        print(f"  Train: Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss/len(val_loader):.4f}, Acc {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs('selective_models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'target_breeds': target_breeds
            }, 'selective_models/best_selective_model.pth')
            
            print(f"✅ Nuevo mejor modelo guardado: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print("⏹️ Early stopping")
            break
    
    print(f"\n🏆 Mejor accuracy de validación: {best_val_acc:.2f}%")
    return 'selective_models/best_selective_model.pth', train_dataset.class_to_idx

if __name__ == "__main__":
    model_path, class_mapping = create_selective_fine_tuning()
    print(f"✅ Modelo selectivo guardado: {model_path}")
    print(f"📋 Clases: {list(class_mapping.keys())}")
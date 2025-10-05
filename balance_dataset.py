#!/usr/bin/env python3
"""
Script de balanceado automático de dataset con data augmentation
"""

import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import json
from pathlib import Path

class DatasetBalancer:
    def __init__(self, dataset_dir, target_images_per_class=161):
        self.dataset_dir = dataset_dir
        self.target_images_per_class = target_images_per_class
        self.backup_dir = f"{dataset_dir}_backup"
        
    def create_backup(self):
        """Crear backup del dataset original"""
        if os.path.exists(self.backup_dir):
            print(f"⚠️ Backup ya existe en: {self.backup_dir}")
            return
            
        print(f"💾 Creando backup en: {self.backup_dir}")
        shutil.copytree(self.dataset_dir, self.backup_dir)
        print("✅ Backup creado exitosamente")
    
    def augment_image(self, image_path, output_path, augmentation_type):
        """Aplicar una transformación específica a una imagen"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                if augmentation_type == 'flip_horizontal':
                    augmented = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                elif augmentation_type == 'rotate_15':
                    augmented = img.rotate(15, expand=True, fillcolor='white')
                    
                elif augmentation_type == 'rotate_-15':
                    augmented = img.rotate(-15, expand=True, fillcolor='white')
                    
                elif augmentation_type == 'brightness_up':
                    enhancer = ImageEnhance.Brightness(img)
                    augmented = enhancer.enhance(1.3)
                    
                elif augmentation_type == 'brightness_down':
                    enhancer = ImageEnhance.Brightness(img)
                    augmented = enhancer.enhance(0.7)
                    
                elif augmentation_type == 'contrast_up':
                    enhancer = ImageEnhance.Contrast(img)
                    augmented = enhancer.enhance(1.3)
                    
                elif augmentation_type == 'contrast_down':
                    enhancer = ImageEnhance.Contrast(img)
                    augmented = enhancer.enhance(0.7)
                    
                elif augmentation_type == 'saturation_up':
                    enhancer = ImageEnhance.Color(img)
                    augmented = enhancer.enhance(1.3)
                    
                elif augmentation_type == 'saturation_down':
                    enhancer = ImageEnhance.Color(img)
                    augmented = enhancer.enhance(0.7)
                    
                elif augmentation_type == 'crop_center':
                    width, height = img.size
                    crop_size = min(width, height) * 0.8
                    left = (width - crop_size) / 2
                    top = (height - crop_size) / 2
                    right = (width + crop_size) / 2
                    bottom = (height + crop_size) / 2
                    augmented = img.crop((left, top, right, bottom))
                    augmented = augmented.resize((width, height))
                    
                else:
                    augmented = img  # Sin cambios
                
                # Guardar imagen aumentada
                augmented.save(output_path, 'JPEG', quality=90)
                return True
                
        except Exception as e:
            print(f"❌ Error augmentando {image_path}: {e}")
            return False
    
    def balance_breed(self, breed_name, current_count):
        """Balancear una raza específica"""
        breed_dir = os.path.join(self.dataset_dir, breed_name)
        
        if current_count > self.target_images_per_class:
            # Reducir imágenes
            needed_reduction = current_count - self.target_images_per_class
            self._reduce_images(breed_dir, needed_reduction)
            print(f"   📉 {breed_name}: {current_count} → {self.target_images_per_class} (-{needed_reduction})")
            
        elif current_count < self.target_images_per_class:
            # Aumentar imágenes con data augmentation
            needed_augmentation = self.target_images_per_class - current_count
            self._augment_images(breed_dir, needed_augmentation)
            print(f"   📈 {breed_name}: {current_count} → {self.target_images_per_class} (+{needed_augmentation})")
        
        else:
            print(f"   ✅ {breed_name}: {current_count} (ya balanceado)")
    
    def _reduce_images(self, breed_dir, reduction_needed):
        """Reducir imágenes manteniendo las de mejor calidad"""
        image_files = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Para simplificar, seleccionar aleatoriamente las imágenes a eliminar
        # En un escenario real, aquí evaluaríamos la calidad de cada imagen
        to_remove = random.sample(image_files, reduction_needed)
        
        for img_file in to_remove:
            img_path = os.path.join(breed_dir, img_file)
            os.remove(img_path)
    
    def _augment_images(self, breed_dir, augmentation_needed):
        """Generar imágenes sintéticas usando data augmentation"""
        original_files = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Técnicas de augmentation disponibles
        augmentation_types = [
            'flip_horizontal',
            'rotate_15',
            'rotate_-15', 
            'brightness_up',
            'brightness_down',
            'contrast_up',
            'contrast_down',
            'saturation_up',
            'saturation_down',
            'crop_center'
        ]
        
        augmented_count = 0
        attempts = 0
        max_attempts = augmentation_needed * 3  # Evitar bucle infinito
        
        while augmented_count < augmentation_needed and attempts < max_attempts:
            # Seleccionar imagen base aleatoria
            base_image = random.choice(original_files)
            base_path = os.path.join(breed_dir, base_image)
            
            # Seleccionar técnica de augmentation aleatoria
            aug_type = random.choice(augmentation_types)
            
            # Generar nombre único para imagen augmentada
            base_name, ext = os.path.splitext(base_image)
            aug_filename = f"{base_name}_aug_{aug_type}_{augmented_count:03d}.jpg"
            aug_path = os.path.join(breed_dir, aug_filename)
            
            # Aplicar augmentation
            if self.augment_image(base_path, aug_path, aug_type):
                augmented_count += 1
            
            attempts += 1
        
        return augmented_count
    
    def balance_full_dataset(self):
        """Balancear todo el dataset"""
        print("🔧 BALANCEADO AUTOMÁTICO DE DATASET")
        print("=" * 50)
        
        # Cargar información de balance
        with open('detailed_balance_report.json', 'r') as f:
            report = json.load(f)
        
        breed_counts = report['analysis']['breed_counts']
        
        print(f"📊 Objetivo: {self.target_images_per_class} imágenes por raza")
        print(f"📁 Procesando {len(breed_counts)} razas...")
        
        # Crear backup
        self.create_backup()
        
        # Procesar cada raza
        for breed_name, current_count in breed_counts.items():
            self.balance_breed(breed_name, current_count)
        
        # Verificar resultado final
        print(f"\n🔍 VERIFICANDO RESULTADO...")
        final_counts = {}
        total_final = 0
        
        for breed_name in breed_counts.keys():
            breed_dir = os.path.join(self.dataset_dir, breed_name)
            final_count = len([f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            final_counts[breed_name] = final_count
            total_final += final_count
        
        # Estadísticas finales
        final_mean = np.mean(list(final_counts.values()))
        final_std = np.std(list(final_counts.values()))
        final_cv = final_std / final_mean
        
        print(f"\n📊 RESULTADO FINAL:")
        print(f"   Total de imágenes: {total_final:,}")
        print(f"   Promedio por raza: {final_mean:.1f}")
        print(f"   Desviación estándar: {final_std:.1f}")
        print(f"   Coeficiente de variación: {final_cv:.3f}")
        
        if final_cv < 0.05:
            print("   🟢 DATASET PERFECTAMENTE BALANCEADO")
        elif final_cv < 0.1:
            print("   🟢 DATASET BIEN BALANCEADO")
        else:
            print("   🟡 DATASET AÚN NECESITA AJUSTES")
        
        # Guardar reporte final
        final_report = {
            'balancing_target': self.target_images_per_class,
            'final_counts': final_counts,
            'final_stats': {
                'total_images': total_final,
                'mean': final_mean,
                'std': final_std,
                'cv': final_cv
            }
        }
        
        with open('balancing_final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n💾 Reporte final guardado en: balancing_final_report.json")
        print(f"💾 Backup original en: {self.backup_dir}")
        
        return final_report

def main():
    """Función principal"""
    
    # Verificar que existe el reporte de análisis
    if not os.path.exists('detailed_balance_report.json'):
        print("❌ Primero ejecuta detailed_balance_analysis.py")
        return
    
    # Configuración
    dataset_dir = "breed_processed_data/train"
    target_per_class = 161  # Objetivo basado en el análisis
    
    # Crear balanceador
    balancer = DatasetBalancer(dataset_dir, target_per_class)
    
    # Ejecutar balanceado
    result = balancer.balance_full_dataset()
    
    print(f"\n✅ BALANCEADO COMPLETADO")
    print(f"🚀 Listo para reentrenar el modelo con dataset balanceado")

if __name__ == "__main__":
    main()
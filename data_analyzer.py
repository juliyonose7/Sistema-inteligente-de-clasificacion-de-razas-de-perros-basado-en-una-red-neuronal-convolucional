"""
Analizador de datasets para clasificación binaria PERRO vs NO-PERRO
Optimizado para GPU AMD 7900XTX con ROCm
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.yesdog_path = self.dataset_path / "YESDOG"
        self.nodog_path = self.dataset_path / "NODOG"
        self.stats = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def analyze_dataset_structure(self):
        """Analiza la estructura completa del dataset"""
        print("🔍 Analizando estructura del dataset...")
        
        # Análisis YESDOG (perros)
        dog_breeds = []
        dog_image_count = 0
        
        for breed_folder in self.yesdog_path.iterdir():
            if breed_folder.is_dir():
                breed_name = breed_folder.name
                images = self._count_images_in_folder(breed_folder)
                dog_breeds.append({
                    'breed': breed_name,
                    'folder': str(breed_folder),
                    'image_count': images
                })
                dog_image_count += images
        
        # Análisis NODOG (objetos)
        nodog_categories = []
        nodog_image_count = 0
        
        for category_folder in self.nodog_path.iterdir():
            if category_folder.is_dir():
                category_name = category_folder.name
                images = self._count_images_in_folder(category_folder)
                nodog_categories.append({
                    'category': category_name,
                    'folder': str(category_folder),
                    'image_count': images
                })
                nodog_image_count += images
        
        self.stats['dog_breeds'] = dog_breeds
        self.stats['nodog_categories'] = nodog_categories
        self.stats['total_dog_images'] = dog_image_count
        self.stats['total_nodog_images'] = nodog_image_count
        self.stats['total_images'] = dog_image_count + nodog_image_count
        self.stats['class_balance'] = {
            'dogs': dog_image_count,
            'no_dogs': nodog_image_count,
            'ratio': dog_image_count / max(nodog_image_count, 1)
        }
        
        print(f"✅ Análisis completado:")
        print(f"   - Razas de perros: {len(dog_breeds)}")
        print(f"   - Categorías no-perro: {len(nodog_categories)}")
        print(f"   - Total imágenes perros: {dog_image_count:,}")
        print(f"   - Total imágenes no-perros: {nodog_image_count:,}")
        print(f"   - Ratio perros/no-perros: {self.stats['class_balance']['ratio']:.2f}")
        
    def _count_images_in_folder(self, folder_path: Path) -> int:
        """Cuenta imágenes válidas en una carpeta"""
        count = 0
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in self.image_extensions:
                count += 1
        return count
    
    def analyze_image_properties(self, sample_size: int = 1000):
        """Analiza propiedades de las imágenes (dimensiones, calidad, etc.)"""
        print(f"📊 Analizando propiedades de imágenes (muestra de {sample_size})...")
        
        image_properties = {
            'widths': [],
            'heights': [],
            'channels': [],
            'file_sizes': [],
            'corrupted': [],
            'aspect_ratios': []
        }
        
        # Muestreo de imágenes de perros
        dog_samples = self._sample_images_from_class('dog', sample_size // 2)
        nodog_samples = self._sample_images_from_class('nodog', sample_size // 2)
        
        all_samples = dog_samples + nodog_samples
        
        for img_path, label in tqdm(all_samples, desc="Analizando imágenes"):
            try:
                # Leer imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    image_properties['corrupted'].append(str(img_path))
                    continue
                
                h, w, c = img.shape
                image_properties['heights'].append(h)
                image_properties['widths'].append(w)
                image_properties['channels'].append(c)
                image_properties['aspect_ratios'].append(w/h)
                
                # Tamaño del archivo
                file_size = Path(img_path).stat().st_size / 1024  # KB
                image_properties['file_sizes'].append(file_size)
                
            except Exception as e:
                image_properties['corrupted'].append(str(img_path))
        
        # Calcular estadísticas
        self.stats['image_properties'] = {
            'width_stats': {
                'mean': np.mean(image_properties['widths']),
                'std': np.std(image_properties['widths']),
                'min': np.min(image_properties['widths']),
                'max': np.max(image_properties['widths']),
                'median': np.median(image_properties['widths'])
            },
            'height_stats': {
                'mean': np.mean(image_properties['heights']),
                'std': np.std(image_properties['heights']),
                'min': np.min(image_properties['heights']),
                'max': np.max(image_properties['heights']),
                'median': np.median(image_properties['heights'])
            },
            'aspect_ratio_stats': {
                'mean': np.mean(image_properties['aspect_ratios']),
                'std': np.std(image_properties['aspect_ratios']),
                'min': np.min(image_properties['aspect_ratios']),
                'max': np.max(image_properties['aspect_ratios'])
            },
            'file_size_stats': {
                'mean_kb': np.mean(image_properties['file_sizes']),
                'median_kb': np.median(image_properties['file_sizes']),
                'min_kb': np.min(image_properties['file_sizes']),
                'max_kb': np.max(image_properties['file_sizes'])
            },
            'corrupted_count': len(image_properties['corrupted']),
            'total_analyzed': len(all_samples)
        }
        
        print(f"✅ Propiedades analizadas:")
        print(f"   - Imágenes corruptas: {len(image_properties['corrupted'])}")
        print(f"   - Dimensión promedio: {self.stats['image_properties']['width_stats']['mean']:.0f}x{self.stats['image_properties']['height_stats']['mean']:.0f}")
        print(f"   - Tamaño promedio: {self.stats['image_properties']['file_size_stats']['mean_kb']:.1f} KB")
        
    def _sample_images_from_class(self, class_type: str, sample_size: int):
        """Muestrea imágenes de una clase específica"""
        images = []
        
        if class_type == 'dog':
            folders = [breed['folder'] for breed in self.stats['dog_breeds']]
        else:
            folders = [cat['folder'] for cat in self.stats['nodog_categories']]
        
        for folder in folders:
            folder_path = Path(folder)
            folder_images = []
            for file in folder_path.iterdir():
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    folder_images.append((file, class_type))
            
            # Muestreo proporcional
            if folder_images:
                n_samples = min(len(folder_images), max(1, sample_size // len(folders)))
                sampled = np.random.choice(len(folder_images), 
                                         size=min(n_samples, len(folder_images)), 
                                         replace=False)
                for idx in sampled:
                    images.append(folder_images[idx])
        
        return images
    
    def create_visualization_report(self):
        """Crea visualizaciones del análisis"""
        print("📈 Creando reporte visual...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis del Dataset PERRO vs NO-PERRO', fontsize=16, fontweight='bold')
        
        # 1. Distribución de clases
        ax1 = axes[0, 0]
        classes = ['Perros', 'No-Perros']
        counts = [self.stats['total_dog_images'], self.stats['total_nodog_images']]
        colors = ['#FF6B6B', '#4ECDC4']
        wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Distribución de Clases')
        
        # 2. Top 10 razas de perros con más imágenes
        ax2 = axes[0, 1]
        dog_df = pd.DataFrame(self.stats['dog_breeds'])
        top_breeds = dog_df.nlargest(10, 'image_count')
        breed_names = [breed.split('-')[-1] for breed in top_breeds['breed']]
        ax2.barh(breed_names, top_breeds['image_count'], color='#FF6B6B', alpha=0.7)
        ax2.set_title('Top 10 Razas (más imágenes)')
        ax2.set_xlabel('Número de imágenes')
        
        # 3. Categorías no-perro
        ax3 = axes[0, 2]
        nodog_df = pd.DataFrame(self.stats['nodog_categories'])
        cat_names = [cat.replace('_final', '') for cat in nodog_df['category']]
        ax3.bar(range(len(cat_names)), nodog_df['image_count'], color='#4ECDC4', alpha=0.7)
        ax3.set_title('Categorías No-Perro')
        ax3.set_xlabel('Categoría')
        ax3.set_ylabel('Número de imágenes')
        ax3.set_xticks(range(len(cat_names)))
        ax3.set_xticklabels(cat_names, rotation=45, ha='right')
        
        # 4. Distribución de dimensiones
        if 'image_properties' in self.stats:
            ax4 = axes[1, 0]
            width_stats = self.stats['image_properties']['width_stats']
            height_stats = self.stats['image_properties']['height_stats']
            
            dimensions = ['Ancho', 'Alto']
            means = [width_stats['mean'], height_stats['mean']]
            stds = [width_stats['std'], height_stats['std']]
            
            ax4.bar(dimensions, means, yerr=stds, capsize=5, color=['#95E1D3', '#F38BA8'], alpha=0.7)
            ax4.set_title('Dimensiones Promedio de Imágenes')
            ax4.set_ylabel('Píxeles')
            
            # 5. Distribución de aspect ratios
            ax5 = axes[1, 1]
            ar_stats = self.stats['image_properties']['aspect_ratio_stats']
            ax5.text(0.1, 0.8, f"Aspect Ratio Promedio: {ar_stats['mean']:.2f}", fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.6, f"Desviación Estándar: {ar_stats['std']:.2f}", fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.4, f"Rango: {ar_stats['min']:.2f} - {ar_stats['max']:.2f}", fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Estadísticas de Aspect Ratio')
            ax5.axis('off')
            
            # 6. Calidad del dataset
            ax6 = axes[1, 2]
            total_analyzed = self.stats['image_properties']['total_analyzed']
            corrupted = self.stats['image_properties']['corrupted_count']
            valid = total_analyzed - corrupted
            
            quality_data = ['Válidas', 'Corruptas']
            quality_counts = [valid, corrupted]
            quality_colors = ['#90EE90', '#FFB6C1']
            
            ax6.pie(quality_counts, labels=quality_data, autopct='%1.1f%%', 
                   colors=quality_colors, startangle=90)
            ax6.set_title('Calidad de Imágenes (Muestra)')
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'dataset_analysis_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Reporte guardado en: {self.dataset_path / 'dataset_analysis_report.png'}")
    
    def generate_recommendations(self):
        """Genera recomendaciones para el preprocesamiento"""
        print("\n💡 RECOMENDACIONES PARA EL MODELO:")
        print("="*50)
        
        # Balance de clases
        ratio = self.stats['class_balance']['ratio']
        if ratio > 2 or ratio < 0.5:
            print(f"⚠️  DESBALANCE DE CLASES detectado (ratio: {ratio:.2f})")
            print("   → Usar técnicas de balanceo (oversampling, undersampling, o class weights)")
        else:
            print(f"✅ Balance de clases aceptable (ratio: {ratio:.2f})")
        
        # Tamaño del dataset
        total = self.stats['total_images']
        if total < 10000:
            print(f"⚠️  Dataset pequeño ({total:,} imágenes)")
            print("   → Usar augmentación agresiva de datos")
            print("   → Considerar transfer learning con modelos preentrenados")
        else:
            print(f"✅ Tamaño de dataset adecuado ({total:,} imágenes)")
        
        # Propiedades de imágenes
        if 'image_properties' in self.stats:
            corruption_rate = self.stats['image_properties']['corrupted_count'] / self.stats['image_properties']['total_analyzed']
            if corruption_rate > 0.01:
                print(f"⚠️  Alto porcentaje de imágenes corruptas ({corruption_rate*100:.1f}%)")
                print("   → Implementar validación robusta de imágenes")
            
            # Recomendaciones de preprocesamiento
            avg_width = self.stats['image_properties']['width_stats']['mean']
            avg_height = self.stats['image_properties']['height_stats']['mean']
            
            print(f"\n📋 PREPROCESAMIENTO RECOMENDADO:")
            print(f"   • Redimensionar a: 224x224 (estándar para transfer learning)")
            print(f"   • Normalización: ImageNet stats [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]")
            print(f"   • Augmentación: rotación (±15°), flip horizontal, crop aleatorio")
            
        print(f"\n🎯 MODELO RECOMENDADO:")
        print(f"   • EfficientNet-B3 o ResNet-50 (preentrenado en ImageNet)")
        print(f"   • Transfer learning: congelar capas iniciales, fine-tune últimas capas")
        print(f"   • Optimizador: AdamW con learning rate scheduling")
        print(f"   • Loss: BCEWithLogitsLoss con class weights si hay desbalance")
        
    def save_analysis_report(self):
        """Guarda el reporte completo en JSON"""
        report_path = self.dataset_path / 'dataset_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Reporte completo guardado en: {report_path}")
        
    def run_complete_analysis(self):
        """Ejecuta el análisis completo"""
        print("🚀 Iniciando análisis completo del dataset...")
        print("="*60)
        
        self.analyze_dataset_structure()
        self.analyze_image_properties(sample_size=2000)
        self.create_visualization_report()
        self.generate_recommendations()
        self.save_analysis_report()
        
        print("\n🎉 ¡Análisis completado exitosamente!")

if __name__ == "__main__":
    # Configurar paths
    dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS"
    
    # Crear analizador y ejecutar
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.run_complete_analysis()
"""
🐕 INICIADOR DE ENTRENAMIENTO BINARIO
Entrena el modelo binario (perro vs no-perro) con control de parada manual
"""

import os
import sys
from binary_trainer import (
    optimize_for_7800x3d, 
    BinaryDogClassifier, 
    BinaryTrainer,
    create_dataloaders,
    get_transforms
)
import torch
from pathlib import Path

def main():
    """Función principal"""
    print("🐕 INICIANDO ENTRENAMIENTO BINARIO CON CONTROL MANUAL")
    print("🚀 Optimizado para AMD 7800X3D")
    print("=" * 80)
    
    # Optimizar para 7800X3D
    optimize_for_7800x3d()
    
    # Configuración
    DATA_PATH = "./DATASETS"
    BATCH_SIZE = 32  # Tamaño de batch optimizado
    NUM_WORKERS = 12  # Para 7800X3D
    
    # Verificar datos
    if not Path(DATA_PATH).exists():
        print(f"❌ Directorio de datos no encontrado: {DATA_PATH}")
        return
    
    # Crear dataloaders
    print("📊 Cargando datasets...")
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = create_dataloaders(
        DATA_PATH, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS
    )
    
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    print()
    
    # Crear modelo
    print("🤖 Creando modelo EfficientNet-B1...")
    model = BinaryDogClassifier(pretrained=True)
    device = torch.device('cpu')  # Usando CPU para consistencia
    
    # Crear trainer
    trainer = BinaryTrainer(model, device=device)
    
    print()
    print("🎯 CONFIGURACIÓN DE ENTRENAMIENTO:")
    print("   - Épocas: 25 (con early stopping)")
    print("   - Paciencia: 5 épocas sin mejora")
    print("   - Optimizador: AdamW con OneCycleLR")
    print("   - Control manual: Presiona 'q' para parar")
    print()
    
    # Entrenar modelo
    results = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=25,
        save_path='./binary_models',
        patience=5
    )
    
    print("🎉 ENTRENAMIENTO COMPLETADO!")
    print("=" * 80)
    print(f"✅ Mejor accuracy: {results['best_accuracy']:.2f}%")
    print(f"📅 Épocas entrenadas: {results['final_epoch']}")
    print(f"💾 Modelo guardado en: ./binary_models/best_binary_model.pth")
    print()
    print("🔄 Para copiar el modelo a la ubicación esperada:")
    print("   copy binary_models\\best_binary_model.pth best_model.pth")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
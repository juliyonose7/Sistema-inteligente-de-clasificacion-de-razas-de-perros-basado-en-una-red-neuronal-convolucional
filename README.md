# 🐕 Dog Classification - Deep Learning Project

Sistema completo de clasificación binaria **PERRO vs NO-PERRO** optimizado para GPU AMD 7900XTX con ROCm.

## 🎯 Características

- **🤖 Modelos preentrenados**: EfficientNet-B3, ResNet-50/101, DenseNet-121
- **⚡ Optimización AMD**: Soporte ROCm para GPU AMD 7900XTX
- **🔧 Optimización de inferencia**: TorchScript, ONNX, Mixed Precision
- **🌐 API REST**: FastAPI con documentación interactiva
- **📊 Análisis completo**: Estadísticas de dataset y visualizaciones
- **🚀 Pipeline automatizado**: Script completo desde datos hasta API

## 📁 Estructura del Proyecto

```
NOTDOG YESDOG/
├── DATASETS/                 # Datasets originales
│   ├── YESDOG/              # 120 razas de perros
│   └── NODOG/               # 12 categorías de objetos
├── data_analyzer.py         # Análisis de datasets
├── data_preprocessor.py     # Preprocesamiento y augmentación
├── model_trainer.py         # Entrenamiento con ROCm
├── inference_optimizer.py   # Optimización para producción
├── api_server.py           # API REST con FastAPI
├── main_pipeline.py        # Pipeline completo
├── config.py               # Configuración del proyecto
├── requirements.txt        # Dependencias
└── README.md              # Este archivo
```

## 🛠️ Instalación

### 1. Configurar ROCm para AMD GPU

```bash
# Verificar que ROCm esté instalado
rocm-smi

# Si no está instalado, seguir guía oficial:
# https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.2
```

### 2. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🚀 Uso Rápido

### Opción 1: Pipeline Automático (Recomendado)

```bash
# Ejecutar pipeline completo
python main_pipeline.py --dataset "./DATASETS"

# Opciones avanzadas
python main_pipeline.py --dataset "./DATASETS" --model "efficientnet_b3" --epochs 50 --balance "undersample"
```

### Opción 2: Paso a Paso

#### 1. Análisis de Datos
```bash
python data_analyzer.py
```

#### 2. Preprocesamiento
```bash
python data_preprocessor.py
```

#### 3. Entrenamiento
```bash
python model_trainer.py
```

#### 4. Optimización
```bash
python inference_optimizer.py
```

#### 5. Iniciar API
```bash
python api_server.py
```

## 🌐 API REST

Una vez iniciado el servidor:

- **URL principal**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

### Endpoints Principales

#### Predicción Individual
```bash
curl -X POST \"http://localhost:8000/predict\" \
  -F \"file=@imagen.jpg\"
```

#### Predicción por Lotes
```bash
curl -X POST \"http://localhost:8000/predict/batch\" \
  -F \"files=@imagen1.jpg\" \
  -F \"files=@imagen2.jpg\"
```

#### Estado del Servicio
```bash
curl \"http://localhost:8000/health\"
```

### Respuesta de Ejemplo

```json
{
  \"success\": true,
  \"prediction\": \"🐕 PERRO\",
  \"probability\": 0.923,
  \"confidence\": \"Alta\",
  \"processing_time_ms\": 45.2,
  \"model_version\": \"torchscript\",
  \"timestamp\": \"2024-01-15T10:30:00\"
}
```

## 📊 Análisis de Dataset

El script de análisis genera:

- **Estadísticas generales**: Número de imágenes por clase
- **Distribución de razas**: Top 10 razas más representadas
- **Propiedades de imágenes**: Dimensiones, tamaños, calidad
- **Recomendaciones**: Preprocesamiento y modelo sugerido
- **Visualizaciones**: Gráficos y reportes en PNG

### Ejemplo de Salida

```
🔍 Analizando estructura del dataset...
✅ Análisis completado:
   - Razas de perros: 120
   - Categorías no-perro: 12
   - Total imágenes perros: 18,624
   - Total imágenes no-perros: 8,350
   - Ratio perros/no-perros: 2.23

💡 RECOMENDACIONES:
   • Usar balanceo de clases (undersample)
   • EfficientNet-B3 con transfer learning
   • Augmentación agresiva de datos
```

## 🎛️ Configuración

Editar `config.py` para personalizar:

### Modelos Disponibles
- `efficientnet_b3` (Recomendado - balance velocidad/precisión)
- `resnet50` (Rápido)
- `resnet101` (Más preciso)
- `densenet121` (Eficiente en memoria)

### Estrategias de Balanceo
- `undersample`: Reduce clase mayoritaria
- `oversample`: Aumenta clase minoritaria
- `none`: Sin balanceo

### Optimización ROCm
```python
ROCM_CONFIG = {
    \"device\": \"cuda\",
    \"mixed_precision\": True,
    \"benchmark\": True
}
```

## 🔧 Optimización para Producción

### TorchScript (Recomendado)
- Mejor para GPU AMD
- Mantiene flexibilidad de PyTorch
- Optimización automática

### ONNX
- Máxima compatibilidad
- Soporte multiplataforma
- Inferencia CPU optimizada

### Benchmark Automático
El sistema compara automáticamente:
- PyTorch original
- TorchScript optimizado  
- ONNX optimizado
- Diferentes tamaños de batch

Ejemplo de resultados:
```
📊 RESULTADOS DEL BENCHMARK
torchscript_b1    :  23.45 ms,  42.6 FPS
pytorch_b1        :  28.12 ms,  35.5 FPS
onnx_b1          :  31.20 ms,  32.1 FPS
🏆 Más rápido: torchscript_b1
```

## 🎯 Resultados Esperados

### Métricas del Modelo
- **Accuracy**: 92-96%
- **Precision**: 90-95%
- **Recall**: 88-94%
- **F1-Score**: 89-94%

### Performance de Inferencia
- **GPU AMD 7900XTX**: 40-60 FPS (batch=1)
- **CPU (fallback)**: 8-15 FPS
- **Memoria GPU**: ~2-4 GB

## 🐛 Troubleshooting

### ROCm no detectado
```bash
# Verificar instalación ROCm
rocm-smi
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Para RX 7900XTX
```

### Error de memoria GPU
- Reducir `batch_size` en `config.py`
- Usar `mixed_precision = True`
- Verificar otras aplicaciones usando GPU

### Dependencias
```bash
# Reinstalar PyTorch con ROCm
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### API no responde
- Verificar que el modelo esté en `./optimized_models/`
- Comprobar puerto 8000 disponible
- Revisar logs en `./logs/app.log`

## 📈 Monitoreo

### Logs
- **Entrenamiento**: Métricas por época, curvas de aprendizaje
- **API**: Requests, tiempo de respuesta, errores
- **Inferencia**: Estadísticas de predicciones

### Archivos Generados
- `dataset_analysis_report.png`: Análisis visual del dataset
- `training_curves.png`: Curvas de entrenamiento
- `sample_visualization.png`: Muestras procesadas
- `model_metadata.json`: Información del modelo

## 🤝 Uso en Frontend

### JavaScript (Fetch API)
```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Predicción:', data.prediction);
    console.log('Probabilidad:', data.probability);
});
```

### Python (Requests)
```python
import requests

files = {'file': open('imagen.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
result = response.json()

print(f\"Resultado: {result['prediction']}\")
print(f\"Probabilidad: {result['probability']:.3f}\")
```

## 📚 Referencias

- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [AMD ROCm Documentation](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para detalles.

---

**Desarrollado para GPU AMD 7900XTX** 🚀
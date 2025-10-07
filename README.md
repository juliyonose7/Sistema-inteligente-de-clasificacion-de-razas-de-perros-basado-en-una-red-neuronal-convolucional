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

##  como acceder a los modelos entrenados con fine tuning y a otros archivos importantes 
por la gran cantidad de espacio que ocupa el proyecto aisle los modelos y otros archivos clave en este link:L
https://drive.google.com/drive/folders/1CIyc2sEuHoAvDhPGuKi-CYh8ZQiphMt0?usp=sharing 
descarga las carpetas y junto al forge puedes acceder al modelo que gustes! (recomiendo el modelo Resnet50 de 119 clases)

## Pantallazos

![WhatsApp Image 2025-10-04 at 15 13 36_20665017](https://github.com/user-attachments/assets/db168037-416c-4c95-8f5b-70b74a7fcd7d)
![WhatsApp Image 2025-10-04 at 15 13 33_18a5771c](https://github.com/user-attachments/assets/4783582d-549b-437a-854e-55f06fcab2e6)
<img width="922" height="1010" alt="image" src="https://github.com/user-attachments/assets/90bf38bb-0c2a-436a-a222-10a57affa3f3" />
<img width="942" height="979" alt="image" src="https://github.com/user-attachments/assets/7ed3d37c-030b-4329-8a51-1f31cbd019ac" />
<img width="834" height="873" alt="image" src="https://github.com/user-attachments/assets/a05b030d-eaa4-4477-aba2-ca59e7e8b5da" />
<img width="812" height="864" alt="image" src="https://github.com/user-attachments/assets/ff4f01a7-c8cc-491d-a0d9-1477879de5dd" />
<img width="913" height="1016" alt="image" src="https://github.com/user-attachments/assets/0f1c3b32-70c8-4b9e-90a4-30845c2b5f37" />
<img width="965" height="483" alt="image" src="https://github.com/user-attachments/assets/aa7a5ec0-37c6-4237-a73a-0688e3c7e8ee" />
<img width="631" height="354" alt="image" src="https://github.com/user-attachments/assets/3dad669f-8920-458d-8375-0e3192f5c87d" />

## 📚 Referencias

- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [AMD ROCm Documentation](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para detalles.

---


**Desarrollado para GPU AMD 7900XTX** 🚀

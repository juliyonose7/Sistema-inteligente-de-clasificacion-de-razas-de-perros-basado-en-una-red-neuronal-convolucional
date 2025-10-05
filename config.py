# Configuración del proyecto
PROJECT_NAME = "Dog Classification API"
VERSION = "1.0.0"

# Rutas de datos
DATASET_PATH = "./DATASETS"
PROCESSED_DATA_PATH = "./processed_data"
MODELS_PATH = "./models"
OPTIMIZED_MODELS_PATH = "./optimized_models"

# Configuración del modelo
MODEL_CONFIG = {
    "model_name": "efficientnet_b3",  # efficientnet_b3, resnet50, resnet101, densenet121
    "input_size": (224, 224),
    "num_classes": 1,
    "pretrained": True
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "freeze_epochs": 5,
    "balance_strategy": "undersample"  # undersample, oversample, none
}

# Configuración de optimización ROCm
ROCM_CONFIG = {
    "device": "cuda",  # cuda para ROCm, cpu para fallback
    "mixed_precision": True,
    "benchmark": True,
    "deterministic": False
}

# Configuración de augmentación
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "rotation_limit": 15,
    "brightness_contrast": 0.2,
    "gaussian_noise": 0.3,
    "cutout": 0.2,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225]
}

# Configuración de la API
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    "max_batch_size": 10
}

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "./logs/app.log"
}
// Configuración de la API
const API_BASE_URL = 'http://localhost:8000';

// Estado de la aplicación
const state = {
    currentImage: null,
    isLoading: false,
    results: null
};

// Elementos del DOM
const elements = {
    uploadArea: null,
    fileInput: null,
    imagePreview: null,
    previewImg: null,
    resetBtn: null,
    loadingContainer: null,
    resultsContainer: null
};

// Inicialización cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    setupEventListeners();
    initParticles();
    checkAPIConnection();
});

// Inicializar referencias a elementos DOM
function initializeElements() {
    elements.uploadArea = document.getElementById('uploadArea');
    elements.fileInput = document.getElementById('fileInput');
    elements.imagePreview = document.getElementById('imagePreview');
    elements.previewImg = document.getElementById('previewImg');
    elements.resetBtn = document.getElementById('resetBtn');
    elements.loadingContainer = document.getElementById('loadingContainer');
    elements.resultsContainer = document.getElementById('resultsContainer');
}

// Configurar event listeners
function setupEventListeners() {
    // Upload area drag and drop
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Reset button
    elements.resetBtn.addEventListener('click', resetInterface);

    // Prevenir comportamiento por defecto del drag
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, preventDefaults);
        document.body.addEventListener(eventName, preventDefaults);
    });
}

// Prevenir comportamientos por defecto
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Manejar dragover
function handleDragOver(e) {
    elements.uploadArea.classList.add('dragover');
}

// Manejar dragleave
function handleDragLeave(e) {
    elements.uploadArea.classList.remove('dragover');
}

// Manejar drop
function handleDrop(e) {
    elements.uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Manejar selección de archivo
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Procesar archivo seleccionado
function processFile(file) {
    // Validar tipo de archivo
    if (!file.type.startsWith('image/')) {
        showError('Por favor selecciona un archivo de imagen válido');
        return;
    }

    // Validar tamaño (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('La imagen es demasiado grande. Máximo 10MB');
        return;
    }

    state.currentImage = file;
    displayImagePreview(file);
    classifyImage(file);
}

// Mostrar preview de la imagen
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        elements.previewImg.src = e.target.result;
        showSection('imagePreview');
        hideSection('uploadArea');
    };
    reader.readAsDataURL(file);
}

// Clasificar imagen usando la API
async function classifyImage(file) {
    try {
        showSection('loadingContainer');
        hideSection('resultsContainer');
        state.isLoading = true;

        const formData = new FormData();
        formData.append('file', file);

        console.log('🔄 Enviando imagen a la API...');
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error de la API:', errorText);
            throw new Error(`Error del servidor: ${response.status} - ${errorText}`);
        }

        const results = await response.json();
        console.log('✅ Respuesta de la API:', results);
        
        // Verificar que la respuesta tenga el formato esperado
        if (!results || !results.success) {
            throw new Error('La API devolvió una respuesta de error');
        }

        state.results = results;
        displayResults(results);

    } catch (error) {
        console.error('❌ Error en clasificación:', error);
        showError(`Error al clasificar la imagen: ${error.message}`);
    } finally {
        state.isLoading = false;
        hideSection('loadingContainer');
    }
}

// Mostrar resultados de clasificación
function displayResults(results) {
    const container = elements.resultsContainer;
    
    // Limpiar contenido anterior
    container.innerHTML = '';

    // Verificar que los datos necesarios estén presentes
    if (!results || !results.top_predictions || !Array.isArray(results.top_predictions)) {
        showError('Error: Formato de respuesta de la API inválido');
        return;
    }

    // Crear card de resultados
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card';

    // Header con resultado principal
    const header = document.createElement('div');
    header.className = 'result-header';
    
    const title = document.createElement('h2');
    title.className = 'result-title';
    // Usar recommendation.most_likely o el primer resultado
    const mainBreed = results.recommendation?.most_likely || results.top_predictions[0]?.breed || 'Desconocido';
    title.textContent = `🐕 ${formatBreedName(mainBreed)}`;
    
    const confidenceBadge = document.createElement('div');
    // Usar recommendation.confidence o el primer resultado
    const mainConfidence = results.recommendation?.confidence || results.top_predictions[0]?.confidence || 0;
    confidenceBadge.className = `confidence-badge ${getConfidenceLevel(mainConfidence)}`;
    confidenceBadge.textContent = `${(mainConfidence * 100).toFixed(1)}% confianza`;
    
    header.appendChild(title);
    header.appendChild(confidenceBadge);

    // Lista de predicciones top
    const predictionsList = document.createElement('div');
    predictionsList.className = 'predictions-list';

    // Mostrar top 5 predicciones
    const topPredictions = results.top_predictions.slice(0, 5);
    topPredictions.forEach((prediction, index) => {
        const item = createPredictionItem(prediction, index + 1);
        predictionsList.appendChild(item);
    });

    resultCard.appendChild(header);
    resultCard.appendChild(predictionsList);
    container.appendChild(resultCard);

    showSection('resultsContainer');
}

// Crear item de predicción
function createPredictionItem(prediction, rank) {
    const item = document.createElement('div');
    item.className = `prediction-item ${rank === 1 ? 'top' : ''}`;

    // Info de la raza
    const breedInfo = document.createElement('div');
    breedInfo.className = 'breed-info';

    const rankBadge = document.createElement('div');
    rankBadge.className = 'breed-rank';
    rankBadge.textContent = rank;

    const breedName = document.createElement('div');
    breedName.className = 'breed-name';
    breedName.textContent = formatBreedName(prediction.breed);

    breedInfo.appendChild(rankBadge);
    breedInfo.appendChild(breedName);

    // Info de confianza
    const confidenceInfo = document.createElement('div');
    confidenceInfo.className = 'confidence-info';

    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'confidence-bar';

    const confidenceFill = document.createElement('div');
    confidenceFill.className = 'confidence-fill';
    confidenceFill.style.width = `${prediction.confidence * 100}%`;

    confidenceBar.appendChild(confidenceFill);

    const confidencePercent = document.createElement('div');
    confidencePercent.className = 'confidence-percent';
    confidencePercent.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    confidenceInfo.appendChild(confidenceBar);
    confidenceInfo.appendChild(confidencePercent);

    item.appendChild(breedInfo);
    item.appendChild(confidenceInfo);

    return item;
}

// Formatear nombre de raza
function formatBreedName(breedName) {
    return breedName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Obtener nivel de confianza
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
}

// Resetear interfaz
function resetInterface() {
    state.currentImage = null;
    state.results = null;
    
    elements.fileInput.value = '';
    elements.previewImg.src = '';
    
    hideSection('imagePreview');
    hideSection('loadingContainer');
    hideSection('resultsContainer');
    showSection('uploadArea');
}

// Mostrar sección
function showSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'block';
    }
}

// Ocultar sección
function hideSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'none';
    }
}

// Mostrar error
function showError(message) {
    // Crear toast de error
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
        font-weight: 500;
    `;
    toast.textContent = message;

    // Agregar estilos de animación
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);

    document.body.appendChild(toast);

    // Remover después de 5 segundos
    setTimeout(() => {
        toast.remove();
        style.remove();
    }, 5000);
}

// Verificar conexión con la API
async function checkAPIConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ Conexión con API establecida');
        } else {
            throw new Error('API no disponible');
        }
    } catch (error) {
        console.warn('⚠️ No se pudo conectar con la API:', error.message);
        showError('No se pudo conectar con el servidor. Asegúrate de que la API esté ejecutándose en el puerto 8000');
    }
}

// Inicializar partículas animadas
function initParticles() {
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    
    // Redimensionar canvas
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Partículas
    const particles = [];
    const particleCount = 50;

    // Crear partículas
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            size: Math.random() * 3 + 1,
            speedX: (Math.random() - 0.5) * 0.5,
            speedY: (Math.random() - 0.5) * 0.5,
            opacity: Math.random() * 0.5 + 0.2
        });
    }

    // Animar partículas
    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(particle => {
            // Actualizar posición
            particle.x += particle.speedX;
            particle.y += particle.speedY;

            // Rebote en bordes
            if (particle.x < 0 || particle.x > canvas.width) particle.speedX *= -1;
            if (particle.y < 0 || particle.y > canvas.height) particle.speedY *= -1;

            // Dibujar partícula
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${particle.opacity})`;
            ctx.fill();
        });

        requestAnimationFrame(animateParticles);
    }

    animateParticles();
}

// Utilidades adicionales
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Función para obtener información del modelo
async function getModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (response.ok) {
            const info = await response.json();
            console.log('📊 Información del modelo:', info);
            return info;
        }
    } catch (error) {
        console.warn('No se pudo obtener información del modelo:', error);
    }
    return null;
}
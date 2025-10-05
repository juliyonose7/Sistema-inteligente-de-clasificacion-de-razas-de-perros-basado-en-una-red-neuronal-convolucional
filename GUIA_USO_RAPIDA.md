# 🚀 Guía de Uso Rápida - Frontend Definitivo

## ⚡ Inicio Rápido (2 pasos)

### 1️⃣ Iniciar API del Modelo
```powershell
cd "c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
python testing_api_119_classes.py
```
**Esperar hasta ver**: `✅ API lista para recibir peticiones!`

### 2️⃣ Iniciar Frontend
```powershell
# Nueva terminal
cd "c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
python start_frontend.py
```
**Se abrirá automáticamente**: http://localhost:3000/simple_frontend_119.html

## 🎯 Uso de la Interfaz

### Subir Imagen
1. **Arrastra** una imagen al área de upload
2. **O haz clic** en "Seleccionar Imagen"
3. **Formatos**: JPG, PNG, WebP, etc.
4. **Tamaño máximo**: 10MB

### Ver Resultados
- 🥇 **Raza principal** con nivel de confianza
- 📊 **Top 5 predicciones** con barras visuales
- 🎨 **Colores adaptativos** según confianza:
  - 🟢 Verde: Alta confianza (>80%)
  - 🟡 Amarillo: Media confianza (50-80%)
  - 🔴 Rojo: Baja confianza (<50%)

### Subir Nueva Imagen
- 🔄 **Clic en "Subir otra imagen"** para resetear
- ✨ **Drag & drop** funciona en cualquier momento

## ⚠️ Problemas Comunes

### "Error al conectar con el servidor"
➡️ **Verificar**: API corriendo en puerto 8000

### "Archivo no válido"
➡️ **Verificar**: Imagen válida y < 10MB

### Página en blanco
➡️ **Verificar**: Frontend corriendo en puerto 3000

## 🔧 Comandos de Emergencia

### Reiniciar Sistema Completo
```powershell
# Ctrl+C en ambas terminales, luego:
python testing_api_119_classes.py    # Terminal 1
python start_frontend.py             # Terminal 2
```

### Verificar Estado
```powershell
# Probar API
curl http://localhost:8000/health

# Probar Frontend
curl http://localhost:3000/simple_frontend_119.html
```

---
💡 **Tip**: Mantén ambas terminales abiertas mientras uses el sistema
🎯 **URL Principal**: http://localhost:3000/simple_frontend_119.html
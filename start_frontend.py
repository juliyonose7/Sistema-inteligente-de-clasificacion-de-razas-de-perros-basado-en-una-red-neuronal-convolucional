#!/usr/bin/env python3
"""
Servidor Simple para Frontend HTML/CSS/JS
Sirve archivos estáticos para el frontend del clasificador de razas de perros
"""

import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import threading
import time

class CORSRequestHandler(SimpleHTTPRequestHandler):
    """Handler con soporte para CORS"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def start_frontend_server(port=3000):
    """Iniciar servidor para frontend"""
    
    # Cambiar al directorio del frontend
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f"🌐 Iniciando servidor frontend en puerto {port}...")
    print(f"📁 Directorio: {frontend_dir}")
    
    try:
        server = HTTPServer(('localhost', port), CORSRequestHandler)
        
        print(f"✅ Servidor frontend iniciado en: http://localhost:{port}")
        print(f"📄 Página principal: http://localhost:{port}/simple_frontend_119.html")
        print("\n🔧 Asegúrate de que la API esté ejecutándose en puerto 8000")
        print("   Ejecuta: python testing_api_119_classes.py")
        print("\n⏹️  Presiona Ctrl+C para detener el servidor")
        
        # Abrir navegador automáticamente después de 2 segundos
        def open_browser():
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}/simple_frontend_119.html")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servidor frontend...")
        server.shutdown()
        server.server_close()
        print("✅ Servidor frontend detenido")
        
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Error: Puerto {port} ya está en uso")
            print(f"💡 Intenta con otro puerto o detén el proceso que usa el puerto {port}")
        else:
            print(f"❌ Error al iniciar servidor: {e}")
        sys.exit(1)

def check_files():
    """Verificar que los archivos necesarios existan"""
    required_files = [
        "simple_frontend_119.html",
        "styles.css",
        "app.js"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ Todos los archivos necesarios están presentes")
    return True

def show_help():
    """Mostrar ayuda"""
    print("""
🐕 Dog Breed Classifier - Frontend Server

Uso:
    python start_frontend.py [puerto]

Argumentos:
    puerto    Puerto para el servidor frontend (default: 3000)

Ejemplos:
    python start_frontend.py          # Puerto 3000
    python start_frontend.py 8080     # Puerto 8080

Archivos necesarios:
    - simple_frontend_119.html (página principal)
    - styles.css (estilos CSS)
    - app.js (lógica JavaScript)

Notas:
    - La API debe estar ejecutándose en puerto 8000
    - El navegador se abrirá automáticamente
    - Usa Ctrl+C para detener el servidor
""")

def main():
    """Función principal"""
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
            return
        
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                print("❌ Error: El puerto debe estar entre 1024 y 65535")
                return
        except ValueError:
            print("❌ Error: El puerto debe ser un número válido")
            return
    else:
        port = 3000
    
    # Verificar archivos
    if not check_files():
        print("\n💡 Asegúrate de ejecutar este script en el directorio con los archivos del frontend")
        return
    
    # Iniciar servidor
    start_frontend_server(port)

if __name__ == "__main__":
    main()
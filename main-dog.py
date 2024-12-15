import cv2
from datetime import datetime
from ultralytics import YOLO
import math
import pyttsx3
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import sys
import threading

# URL RTSP de tu cámara
camera_url = "rtsp://admin-admin:Kayn!%25123a@192.168.1.12:554/stream1"

# Abrir el flujo de video
cap = cv2.VideoCapture(camera_url)
if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara.")
    exit()

# Resolución deseada
desired_width = 1280
desired_height = 720

# Definir la línea virtual
start_point = (int(desired_width * 0.3), int(desired_height * 0.4))  # Coordenadas de inicio
end_point = (int(desired_width * 0.7), int(desired_height * 0.4))    # Coordenadas de fin

# Cargar modelos YOLO
model_dog = YOLO("yolov8n.pt")  # Modelo entrenado solo para perros

# Configuración del motor de texto a voz
engine = pyttsx3.init()

# Diccionario para rastrear cruces recientes
crossed_objects = set()

# Funciones auxiliares para texto a voz
def speak(text):
    """Ejecuta el motor de texto a voz en un hilo separado."""
    def run():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()

class DetectionThread(QThread):
    frame_processed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        global crossed_objects

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Redimensionar el cuadro
            frame_resized = cv2.resize(frame, (desired_width, desired_height))

            # Realizar detección de perros con YOLO
            results_dog = model_dog(frame_resized)

            # Dibujar las detecciones en el frame
            for result in results_dog:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    if confidence > 0.70:  # Umbral de confianza
                        # Obtener coordenadas de la caja
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Dibujar caja y etiqueta
                        label = f"Dog {math.ceil(confidence * 100)}%"
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Verificar si la caja cruza la línea
                        if y2 > start_point[1] - 5 and y1 < start_point[1] + 5:  # Tolerancia de ±5 píxeles
                            object_id = (x1, y1, x2, y2)  # Identificador único basado en coordenadas
                            if object_id not in crossed_objects:
                                crossed_objects.add(object_id)  # Registrar objeto
                                current_time = datetime.now().strftime('%H:%M:%S')
                                print(f"¡Perro cruzó la línea! Hora: {current_time}")
                                speak("Un perro ha cruzado la línea")

            # Dibujar la línea verde
            cv2.line(frame_resized, start_point, end_point, (0, 255, 0), 3)

            # Emitir el frame procesado
            self.frame_processed.emit(frame_resized)

    def stop(self):
        self.running = False
        self.wait()

class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detección de Cruces")
        self.setGeometry(100, 100, desired_width, desired_height)

        # Configurar el widget principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout principal
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Label para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Hilo de detección
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_processed.connect(self.update_frame)
        self.detection_thread.start()

    def update_frame(self, frame):
        # Convertir el frame a formato compatible con PyQt5
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        step = channel * width
        q_image = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)

        # Actualizar el QLabel con el frame
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.detection_thread.stop()
        cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec_())

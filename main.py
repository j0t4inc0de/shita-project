import cv2
from datetime import datetime
from ultralytics import YOLO
import math
import pyttsx3

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
model_dog = YOLO("best.pt")  # Modelo entrenado solo para perros

# Configuración del motor de texto a voz
engine = pyttsx3.init()

# Diccionario para rastrear cruces recientes
crossed_objects = set()

print("Mostrando video con detección de cruce...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el flujo de video.")
        break

    # Redimensionar el cuadro
    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    # Realizar detección de perros con YOLO
    results_dog = model_dog(frame_resized)

    # Procesar resultados
    for result in results_dog:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence > 0.70:  # Umbral de confianza
                # Obtener coordenadas de la caja
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calcular centro del objeto
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Dibujar caja y etiqueta
                label = f"Dog {math.ceil(confidence * 100)}%"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Verificar cruce de la línea
                if center_y > start_point[1] - 5 and center_y < start_point[1] + 5:
                    object_id = (x1, y1, x2, y2)  # Identificador único basado en coordenadas
                    if object_id not in crossed_objects:
                        crossed_objects.add(object_id)  # Registrar objeto
                        current_time = datetime.now().strftime('%H:%M:%S')
                        print(f"¡Perro cruzó la línea! Hora: {current_time}")
                        engine.say("Un perro ha cruzado la línea")
                        engine.runAndWait()

    # Dibujar la línea verde
    cv2.line(frame_resized, start_point, end_point, (0, 255, 0), 2)

    # Mostrar el video
    cv2.imshow("Detección de cruces", frame_resized)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

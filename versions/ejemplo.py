import math
import cv2
from ultralytics import YOLO

# Iniciar cámara
camera_url = "rtsp://admin-admin:Kayn!%25123a@192.168.1.12:554/stream1"
cap = cv2.VideoCapture(camera_url)

# Ajustar tamaño
cap.set(3, 1080)
cap.set(4, 720)

# Cargar el modelo YOLO
model = YOLO('best.pt')

# Obtener las clases del modelo cargado
classNames = model.names  # Esto devuelve un diccionario como {0: 'dog'}

# Bucle para mostrar la cámara
while True:
    # Capturar la cámara
    success, img = cap.read()

    # Detectar los objetos
    results = model(img, stream=True)

    # Detección mediante un for para recorrer los resultados
    for r in results:
        # Obtener las cajas de los objetos detectados
        boxes = r.boxes
        for b in boxes:
            # Porcentaje de confianza del objeto detectado
            conf = math.ceil(b.conf[0] * 100)
            if conf > 70:  # Filtrar detecciones con confianza mayor a 70%
                # Obtener las coordenadas para dibujar la caja
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # Dibujar rectángulo en la imagen
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                # Obtener el nombre de la clase detectada
                class_id = int(b.cls[0])  # ID de la clase detectada
                nombre = classNames[class_id]  # Nombre de la clase
                # Añadir texto en la imagen
                cv2.putText(
                    img, f'{nombre} {conf}%', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2
                )

    # Mostrar la cámara en pantalla
    cv2.imshow('Webcam', img)

    # Definir una tecla para cerrar la cámara
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara
cap.release()
# Cerrar ventana
cv2.destroyAllWindows()

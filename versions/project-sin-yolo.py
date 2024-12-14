import cv2
from datetime import datetime

# URL RTSP de tu cámara
camera_url = "rtsp://admin-admin:Kayn!%25123a@192.168.1.12:554/stream1"

# Abrir el flujo de video
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara.")
    exit()

# Resolución deseada
desired_width = 640
desired_height = 360

# Definir las coordenadas de la línea como un rango
line_y = int(desired_height * 0.35)  # 35% del alto
line_tolerance = 2  # Tolerancia de ±2 píxeles
line_start_x = int(desired_width * 0.3)  # 30% del ancho
line_end_x = int(desired_width * 0.7)  # 70% del ancho

# Crear el sustractor de fondo con parámetros ajustados
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Variables para rastrear objetos que cruzan la línea
crossed_objects = set()  # Almacena identificadores únicos de objetos que ya cruzaron

print("Mostrando video con detección de cruce...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el flujo de video.")
        break

    # Redimensionar el cuadro al tamaño deseado
    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    # Aplicar el sustractor de fondo
    fgmask = fgbg.apply(frame_resized)

    # Limpiar la máscara con operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Encontrar los contornos de los objetos en movimiento
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos encontrados
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filtrar áreas pequeñas (ajustado para reducir ruido)
            # Obtener el rectángulo delimitador del contorno
            (x, y, w, h) = cv2.boundingRect(contour)

            # Calcular el centro del objeto
            object_center = (x + w // 2, y + h // 2)

            # Verificar si el objeto cruza la línea
            # Condición 1: El objeto debe estar dentro del rango horizontal de la línea
            if line_start_x <= object_center[0] <= line_end_x:
                # Condición 2: El objeto debe cruzar la línea de arriba hacia abajo o viceversa
                if (y < line_y and y + h > line_y) or (y > line_y and y + h < line_y):
                    if (x, y, w, h) not in crossed_objects:  # Verificar que el objeto no haya sido registrado antes
                        current_time = datetime.now().strftime('%H:%M:%S')
                        print(f"¡Objeto cruzó la línea! Hora: {current_time}")
                        crossed_objects.add((x, y, w, h))

            # Dibujar el contorno del objeto
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Dibujar la línea en el video
    cv2.line(frame_resized, (line_start_x, line_y), (line_end_x, line_y), (0, 255, 0), 2)

    # Mostrar el video redimensionado
    cv2.imshow("Video con detección de cruce", frame_resized)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

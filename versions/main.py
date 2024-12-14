import cv2
from datetime import datetime

# URL RTSP de tu cámara
camera_url = "rtsp://admin-admin:Kayn!%25123a@192.168.1.12:554/stream1"

# Abrir el flujo de video
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara.")
    exit()

# Resolución deseada (por ejemplo, Full HD)
desired_width = 1280
desired_height = 720

# Definir las coordenadas de la línea
start_point = (int(desired_width * 0.3), int(desired_height * 0.4))  # 30% del ancho, 40% del alto
end_point = (int(desired_width * 0.7), int(desired_height * 0.4))    # 70% del ancho, 40% del alto

# Crear el sustractor de fondo
fgbg = cv2.createBackgroundSubtractorMOG2()

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

    # Encontrar los contornos de los objetos en movimiento
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos encontrados
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar áreas pequeñas (ajustar según necesidad)
            # Obtener el rectángulo delimitador del contorno
            (x, y, w, h) = cv2.boundingRect(contour)

            # Verificar si el contorno cruza la línea
            if y + h > start_point[1] and y < end_point[1]:
                # Obtener la hora y los segundos actuales
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"¡Objeto cruzó la línea! Hora: {current_time}")

            # Dibujar el contorno y la línea
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Dibujar la línea en el video
    cv2.line(frame_resized, start_point, end_point, (0, 255, 0), 2)

    # Mostrar el video redimensionado
    cv2.imshow("Video con detección de cruce", frame_resized)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

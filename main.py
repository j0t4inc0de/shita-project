import math
import cv2
from ultralytics import YOLO

#   iniciar camara
camera_url = "rtsp://admin-admin:Kayn!%25123a@192.168.1.12:554/stream1"

# Abrir el flujo de video
cap = cv2.VideoCapture(camera_url)
#   ajustar tamaño
cap.set(3, 1080)
cap.set(4, 720)

#   modelo de YOLO
model = YOLO('yolov8n.pt')
#   clases de YOLO pre entrenadas
classNasme = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
            ]

#   bucle para mostrar la camara
while True:
    #   capturar la camara
    succes, img = cap.read()

    #   detectar los objetos
    results = model(img, stream=True)
    print(results)
    #   deteccion mediante un for para recorrer los resultados
    for r in results:
        print(f"\t{r}")
        #   obtenemos la caja para mostrarla luego en pantalla
        box = r.boxes
        #   recorremos la caja 'box'
        for b in box:
            #   porcentaje del objeto detectado
            porcentaje = math.ceil((box.conf[0]*100))
            print(porcentaje)
            if porcentaje > 70:
                #   obtener las cordenadas para dibujar la BOX
                x1, y1, x2, y2 = b.xyxy[0]
                #   convertimos a entero los valores
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #   rectangulo en la camara
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,0), 3)
                #   detallar el objeto detectado
                org = [x1,x2] # cordenadas
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL # fuente
                fontScale = 1
                color = (255,255,255)
                ancho = 2
                # capturamos clase individual
                nombre = classNasme[int(b.cls[0])]
                #   añadimos texto en la imagen
                cv2.putText(img,f'{nombre} {porcentaje}', org, font, fontScale, color, ancho)
    #   mostrar la camara en pantalla
    cv2.imshow('Webcam', img)
    #   definimos una tecla para cerrar la camara
    if cv2.waitKey(1) == ord('q'):
        break

#   liberar la camara
cap.release()
#   cerrar ventana
cv2.destroyAllWindows()
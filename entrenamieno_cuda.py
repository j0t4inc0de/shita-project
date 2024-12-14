import torch
print(torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
import time
print("1s"), time.sleep(1)
print("2s"), time.sleep(1)

#   Entrenamiento
print('Estado de CUDA:', torch.cuda.is_available())
print('Version:', torch.version.cuda)
print('Numero de dispositivos:', torch.cuda.device_count())
print('Dispositivos:', torch.cuda.get_device_name(0))
print(f"Memoria disponible en la GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="JHs0I92NKOAPCY1PuJsw")
project = rf.workspace("patton-hunter").project("test-qcw5e")
version = project.version(1)
dataset = version.download("yolov8")
                
# configurar ruta del datasets
data_yaml = 'C:\\Users\\jeric\\OneDrive\\Documents\\shita-project\\test-1\\data.yaml'

from ultralytics import YOLO
# crear modelo YOLO
model = YOLO('yolov8n.pt')
resultado = model.train(
    data = data_yaml,
    epochs = 100,
    imgsz = 416,
    batch = 2,
    device = 'cuda',
    project = 'entrenamiento',
    name = 'yolo_custom',
    amp=False  # Desactiva el entrenamiento con precisión mixta
)
validar = model.val()

model.export(format='onnx')

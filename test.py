from ultralytics import YOLO
import torch

print("CUDA:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")

print("YOLO Ready")
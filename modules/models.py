import cv2
import ultralytics
from ultralytics import YOLO

model = ultralytics.YOLO('yolov8s.pt')
traffic_vid = cv2.VideoCapture(0)
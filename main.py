from ultralytics import YOLO
import cv2

model = YOLO(model="yolomodels/yolov8l.pt")

results = model("images/101116traffic3_960x540.jpg",show=True ,)

cv2.waitKey(0)
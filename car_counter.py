from ultralytics import YOLO
import cvzone
import cv2
import math
from sort import *
# Create a resizable window
cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)

# Load YOLO model
model = YOLO("yolomodels\yolov8l.pt")

# Uploading Files
cap = cv2.VideoCapture("videos/cars7.mp4")
masked = cv2.imread("images/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [385, 310, 674, 310]
totalCount = []

#class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(src1=img ,src2=masked)    
    if not success:
        break
        
    # Run YOLO model 
    result = model(imgRegion, stream=True)
    
    detections = np.empty((0,5))
    
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1


            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass in ["car", "motorbike", "bus", "truck"] and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections=np.vstack((detections , currentArray))




    resultTracker = tracker.update(detections)
    
    cv2.line(img , (limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultTracker:
        x1, y1, x2, y2,id = result
        print(result)
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.cornerRect(img,(x1, y1, w, h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-20 <cy<limits[1]+20:
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(img , (limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
    
    
    cvzone.putTextRect(img , f'count :{len(totalCount)}',(50,50))           
    cv2.imshow("image",img)
    # cv2.imshow("mask",imgRegion)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

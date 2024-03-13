from ultralytics import YOLO
import cvzone
import cv2
import math
from sort import *
# Create a resizable window
cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)

# Load YOLO model
model = YOLO("yolomodels\yolov8n.pt")

# Uploading Files
cap = cv2.VideoCapture("videos/people.mp4")

masked = cv2.imread("images/people_counter_mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []

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
            if currentClass =="person" and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections=np.vstack((detections , currentArray))




    resultTracker = tracker.update(detections)
    
    cv2.line(img , (limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,0),3)
    cv2.line(img , (limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,0),3)
    for result in resultTracker:
        x1, y1, x2, y2,id = result
        print(result)
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.cornerRect(img,(x1, y1, w, h),l=9,rt=1,colorR=(255,0,255))
        # cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=2,offset=10)
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-20 <cy<limitsUp[1]+20:
            if totalCountUp.count(id)==0:
                totalCountUp.append(id)
                cv2.line(img , (limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),3)
        
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-20 <cy<limitsDown[1]+20:
            if totalCountDown.count(id)==0:
                totalCountDown.append(id)
                cv2.line(img , (limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,0),3)
    
    
    cvzone.putTextRect(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cvzone.putTextRect(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)           
    cv2.imshow("image",img)
    # cv2.imshow("mask",imgRegion)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
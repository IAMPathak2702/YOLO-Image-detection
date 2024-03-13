# YOLO Image Detection Projects 🖼️

## Introduction ℹ️
This repository contains implementations of various image detection projects using YOLO (You Only Look Once) object detection algorithm. YOLO is a state-of-the-art, real-time object detection system that is extremely fast and accurate. The projects included in this repository cover different applications such as car counting 🚗, people counting 👥, Personal Protective Equipment (PPE) detection 👷, and Poker hand detection ♠️.

## Projects 🚀
1. **Car Counter** 🚗
   - **Description**: This project aims to detect and count cars in images or video streams. It can be used for traffic monitoring, parking lot occupancy estimation, and other related applications.
   - **Model Used**: YOLOv8l.pt
   - **Details**: The YOLOv8l model is utilized for car detection. It's trained on a dataset containing annotated images of cars, enabling it to accurately detect cars in various contexts.

2. **People Counter** 👥
   - **Description**: The people counter project focuses on detecting and counting people in images or video feeds. It can be utilized for crowd management, event monitoring, and safety enforcement purposes.
   - **Model Used**: YOLOv8n.pt
   - **Details**: YOLOv8n model is employed for people detection. This model is optimized for speed and efficiency while maintaining high accuracy. It's trained on a dataset specifically annotated for human detection.

3. **PPE Detection (Custom Training)** 👷
   - **Description**: This project involves detecting whether people in images or videos are wearing Personal Protective Equipment (PPE) such as helmets, vests, goggles, etc. It's crucial for workplace safety compliance and monitoring.
   - **Model Used**: Custom-trained YOLO model (yolov8n.pt)
   - **Details**: The YOLOv8n model is fine-tuned on a custom dataset containing annotated images of individuals wearing various types of PPE. This fine-tuning process enhances the model's ability to accurately detect PPE items in different scenarios.

4. **Poker Hand Detection** ♠️
   - **Description**: The Poker hand detection project is designed to recognize and classify different poker hands in images or video frames. It can be used for gaming applications, player assistance, and analysis.
   - **Model Used**: YOLOv8l.pt
   - **Details**: YOLOv8l model is employed for detecting poker hands. The model is trained on a dataset containing annotated images of various poker hands. It's capable of accurately identifying different combinations of cards in real-time.

## Requirements 🛠️
- Python 3.x
- PyTorch
- OpenCV
- YOLOv8l.pt (for car counter and poker hand detection)
- YOLOv8n.pt (for people counter and PPE detection)

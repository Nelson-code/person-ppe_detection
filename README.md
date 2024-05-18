# person-ppe_detection

**Objective:** This repository is focused towards building an intelligent system using ultrlytics YOLOv8 pretrained model for detecting the persons and personal protective equipments from the given image

# Problem statement

1. Python script to convert the annotations from PascalVOC format to yolov8 format.
2. Train yolov8 object detection model for person detection.
3. Train another yolov8 object detection model for PPE detection (hard-hat, gloves, mask, glasses, boots, vest, ppe-suit, ear-protector, safety-harness)
4. Write the flow which will take an image directory as input, perform inference through both the models and save them in another directory (inference.py).

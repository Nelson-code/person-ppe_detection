#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import required libraries
import argparse as ag
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# In[12]:


def annotate(predict,img,output_path,flag,count):
    """""
    parameters:
    predict= Info about Model predictions including bbox co-ordinates, classes, confidence score
    output_dir=directory to store output images
    flag= to denote the type of model (1: ppe_detection_model, 2: person_detection_model)
    count: counter value for generating unique output file_name for each image

    """""
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.5
    color = (255, 255, 0) 
    thickness = 1
    if(flag==1):
        class_name={0: 'boots', 1: 'gloves', 2: 'hat', 3: 'ppe-suit', 4: 'vest'}
        f_name="ppe_detect"
    else:
        class_name={0: 'person'}
        f_name="person"
    
    annotations=predict.boxes.xyxy
    classes=list(map(int,predict.boxes.cls))
    conf_score=list(map(float,predict.boxes.conf))
    conf_score=[round(ind*100,2) for ind in conf_score]
    for ind,ann in enumerate(annotations):
        l,r,t,b=list(map(int,ann))
        cv2.rectangle(img,(l,r),(t,b),(0,0,255),2) #to draw the rectangle based on the x_min,y_min,x_max,y_max
        if(r-10<0):
            l=l-10
        else:
            r=r-10
        cv2.putText(img,f"{class_name[classes[ind]]}{conf_score[ind]}",(l,r), font,fontScale, color, thickness,cv2.LINE_AA) #to insert text on the given image that includes classes, confidence scores
    
    filename = os.path.join(output_path, f"image_{count}.jpg")
    cv2.imwrite(filename,img)


# In[13]:


def input1(input_dir,output_dir,person_detetctor,ppe_detector):
    """""
    parameters:
    input_dir= path of the input directory
    output_dir=directory to store output images
    person_detetctor= person_detetction_model
    ppe_detetctor= ppe_detetction_model

    """""
    count=0
    for name in os.listdir(input_dir): #extract images from the given input_dir
        img_path=os.path.join(input_dir,name)
        img=cv2.imread(img_path)
        count+=1
        predict1=person_detetctor.predict(img)[0]
        predict2=ppe_detector.predict(img)[0]
        img2=img.copy()
        
        #function to draw bbox and insert text on the image based on the model's predictions
        annotate(predict2,img,output_path,1,count)
        annotate(predict1,img2,output_path,0,count)


# In[ ]:


if __name__ == "__main__":
    
    ap_obj=ag.ArgumentParser(description="Model Prediction") # set up an ArgumentParser object
    # add a positional arguments to the parser.
    ap_obj.add_argument("input_dir",type=Path,help="Directory containing input files")
    ap_obj.add_argument("output_dir",type=Path,help="Directory to store output files")
    ap_obj.add_argument("person_model",type=Path,help="Person_detector_file")
    ap_obj.add_argument("ppe_model",type=Path,help="PPE_detector_file")
    args=ap_obj.parse_args()
    model_path1=args.person_model
    model_path2=args.ppe_model
    person_detector=YOLO(model_path1)
    ppe_detector=YOLO(model_path1)
    
    #Function to predictions on the images in the given input_dir
    input1(args.input_dir,args.output_dir,person_detetctor,ppe_detector)


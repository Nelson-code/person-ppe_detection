#!/usr/bin/env python
# coding: utf-8

# In[47]:


#import the required libraries
import argparse as ag
from pathlib import Path
import xml.etree.ElementTree as ET


# In[49]:


def convert_to_yolo(box,size):
    """""
    parameters:
    
    box: tuple containing bounding box informations
    size: tuple containing image width & height of the given image
    """""

    dw=size[0]
    dh=size[1]
    
    #Calculate the relateive co-ordinates of center, relative height, relative width of the bouding box
    rect_x_center=((box[0]+box[2])/2.0)/dw
    rect_y_center=((box[1]+box[3])/2.0)/dh
    rect_width=(box[2]-box[0])/dw
    rect_height=(box[3]-box[1])/dh
    
    #return the YOLO Bounding Box
    return (rect_x_center,rect_y_center,rect_width,rect_height)


# In[51]:


def xml_text(x_file,output_file,classes):
    """""
    parameters:
    
    x_file: xml file that contains annotations in PascalVOC format
    output_file: text file that will be containing annotations in YOLOv8 format
    classes: list of unique classes used for class mapping
    
    """""
    
    #code block to parse if x_file is of .txt type
    if x_file.suffix==".txt":
        try:
            with x_file.open("r",encoding="utf-8") as file:
                content=file.read()
                root=ET.fromstring(content)
        except ET.parseError as e:
            print(f"Error parsing {x_file}: {e}")
            return
    
    #code block to parse if x_file is of .xml type
    else:
        try:
            tree=ET.parse(x_file)
            root=tree.getroot()
        except ET.parseError as e:
            print("Error parsing {x_file}: e")
    
    #retrieve the image dimesnsion information
    size_ele=root.find("size") #find the size element to extract the image height&width information
    img_width=int(size_ele.find("width").text)
    img_height=int(size_ele.find("height").text)
    
    #Open a .txt file in write mode to store the bounding box coordinates in YOLOv8 format
    with output_file.open('w') as file:
        for obj in root.iter("object"): # iterate through each object element incase of multiple classes in the given image
            c_name=obj.find("name").text
            if c_name not in classes:
                continue
            cl_id=classes.index(c_name)
            
            #retrieve the bounding box informations and store it a tuple of integer type
            dim=obj.find("bndbox")
            box=(float(dim.find("xmin").text),
            float(dim.find("ymin").text),
            float(dim.find("xmax").text),
            float(dim.find("ymax").text))
            
            #function to converts bounding box coordinates from PASCAL VOC format to YOLO format.
            y_box=convert_to_yolo(box,(img_width,img_height))
            
            #writes bound box informations along with class_id(s) into the text file
            file.write(f"{cl_id} {' '.join(map(str,y_box))}\n")


# In[52]:


def process_fn(input_dir,output_dir,classes_file):
    """""
    parameters
    1. input_dir: path of input directory containing xml files
    2. output_dir: path of output directory to store files containing yoloV8 annotations
    3. classes_file: path to class (or) labels
    
    """""
    
    classes=classes_file.read_text().splitlines()
    output_dir.mkdir(parents=True,exist_ok=True) # create a output directory
    
    #Generate a .txt file (YOLO formatted) for every XML file
    for xml in input_dir.glob("*"):
        o_txt_path=output_dir/xml.with_suffix(".txt").name
        
        #Function to convert xml to convert text file
        xml_text(xml,o_txt_path,classes)


# In[27]:


if __name__ == "__main__":
    
    ap_obj=ag.ArgumentParser(description="Conversion of PascalVOC to YoloV8 format") # set up an ArgumentParser object
    # add a positional arguments to the parser.
    ap_obj.add_argument("input_dir",type=Path,help="Directory containing input files")
    ap_obj.add_argument("output_dir",type=Path,help="Directory containing output files")
    ap_obj.add_argument("classes_file",type=Path,help="File containing classes")
    args=ap_obj.parse_args()
    
    #Function to process the xml files and class file
    process_fn(args.input_dir,args.output_dir,args.classes_file)


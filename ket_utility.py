#Utility functions

import csv
import numpy as np
import cv2

#possible reduction in zero steering images
def reduce_zero_steering(input_lines, z_pro=0.2):
    new_lines=[]
    for each_line in input_lines:
        if ((float(each_line[3])==0.0) and (np.random.rand(1)<z_pro)) or (float(each_line[3]) !=0.0):
            new_lines.append(each_line)    
    return new_lines

def purge_zero_speed(input_lines):
    new_lines=[]
    for each_line in input_lines:
        if ((float(each_line[6])>=0.1)):            
            new_lines.append(each_line)    
    return new_lines
    
#generator
## log file is located './folder/driving_log.csv'
## IMG file is located './folder/IMG'
def get_lines(datafolders, split_char_):
    lines=[]
    for i in range(len(datafolders)):
        logfile=datafolders[i]+'driving_log.csv'
        imfile=datafolders[i]+'IMG/'
        with open(logfile) as file_:
            reader=csv.reader(file_)
            for line in reader:
                #center, left, right
                for j in range(3):
                    file_name=line[j].split(split_char_)[-1]
                    current_path= imfile + file_name
                    line[j]=current_path
                    
                lines.append(line)
                
    return lines

def get_left_right_aug(input_lines):
    #line=image_path, steering_angle, augment=True,False
    #Augment True, no flipping
    #Augment Fasle, flip the image in generator
    lines=[]
    for line in input_lines:
        templine=[]
        #center
        templine.append(line[0])
        templine.append(float(line[3]))
        templine.append(True)
        lines.append(templine)
        #left +0.25
        templine=[]

        templine.append(line[1])
        templine.append(float(line[3])+0.25)
        templine.append(True)
        lines.append(templine)       
        #right
        templine=[]

        templine.append(line[2])
        templine.append(float(line[3])-0.25)
        templine.append(True)
        lines.append(templine)      

        if abs(float(line[3])) >= 0.3:
            #center flip
            #center
            templine=[]

            templine.append(line[0])
            templine.append(-1.0*float(line[3]))
            templine.append(False)
            lines.append(templine)
            #left +0.25
            templine=[]

            templine.append(line[1])
            templine.append(-1.0*(float(line[3])+0.25))
            templine.append(False)
            lines.append(templine)       
            #right
            templine=[]
            templine.append(line[2])
            templine.append(-1.0*(float(line[3])-0.25))
            templine.append(False)
            lines.append(templine)
            
    return lines

def process_image(image):
    image=cv2.GaussianBlur(image, (3,3), 0)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


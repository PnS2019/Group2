import os
import cv2 as cv
import numpy as np
import copy
          

files= os.listdir('resized')

photos=[]
resized_photos=[]


upper_bound_rectangle_area=750
lower_bound_rectangle_area=30



#defining a sort function
def xCoo(arr):
    return arr[0][0]


#loading files
for file in files[:]:
    if file.endswith('jpg'):
        photos.append(cv.imread('resized\\'+file))




#color detection

lower_blue = np.array([100,40,0])
upper_blue = np.array([135,255,255])



Blue_filtered=[]

    
for photo in photos:
    
    #converting colorspace:
    hsv = cv.cvtColor(photo,cv.COLOR_BGR2HSV)

    #filtered version of Picture:
    filtered = cv.inRange(hsv,lower_blue,upper_blue)
    filtered = cv.bitwise_not(filtered)
   
    
    Blue_filtered.append(filtered)



#contour detection:


for picture in Blue_filtered:
    
    _,ctrs, hier = cv.findContours(picture, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    picture=cv.cvtColor(picture,cv.COLOR_GRAY2RGB)

    #creting an array of rectangles

    rects = [cv.boundingRect(ctr) for ctr in ctrs]

    #calculating the area of an rectangle
    areas = [cv.contourArea(ctres) for ctres in ctrs]

    
    #combining area and rectangle into one array
    rects_and_area=[]

    j=0
    for itts in rects:
        rects_and_area.append([rects[j],areas[j]])
        j+=1

    #sorting the areas and rectangles from left to right 
    rects_and_area.sort(key=xCoo)

            

    
            
    untouched= copy.deepcopy(picture)

    for rect in rects_and_area:
        if(rect[1]<upper_bound_rectangle_area and rect[1]>lower_bound_rectangle_area):
                       
            cv.rectangle(untouched, (rect[0][0],rect[0][1]) , (rect[0][0]+rect[0][2],rect[0][1]+rect[0][3]), (0, 255, 0), 3)
                    


                        
    cv.imshow('untouched',untouched)
    cv.waitKey(0)

    

    
    for rectangle in rects_and_area:

        
            
        if(rectangle[1]<upper_bound_rectangle_area and rectangle[1]>lower_bound_rectangle_area):
            offset=3
            if (rectangle[0][1]-offset>=0 and
                rectangle[0][1]+rectangle[0][3]+offset<=int(picture.shape[1]) and
                rectangle[0][0]-offset>=0 and
                rectangle[0][0]+rectangle[0][2]+offset<=int(picture.shape[0])):
                
                out_cropped=picture[rectangle[0][1]-offset:rectangle[0][1]+rectangle[0][3]+offset,
                                    rectangle[0][0]-offset:rectangle[0][0]+rectangle[0][2]+offset]
            else:
                out_cropped=picture[rectangle[0][1]:rectangle[0][1]+rectangle[0][3],
                                    rectangle[0][0]:rectangle[0][0]+rectangle[0][2]]
            out_cropped=cv.resize(out_cropped,(200,200))
                
            cv.imshow('cropped',out_cropped)
            cv.waitKey(0)
                
                
                
            

                
        

   













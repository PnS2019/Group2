import os
import cv2 as cv
import numpy as np
import copy
import string

files= os.listdir('original')

photos=[]
resized_photos=[]


upper_bound_rectangle_area=18000
lower_bound_rectangle_area=500

letter_count = [0 for i in range(26)]
for image in os.listdir("label_HighRes"):
    letter = image.split("_")[0]
    number = int(image.split("_")[1].split(".")[0])+1
    index = string.ascii_lowercase.index(letter)
    if number > letter_count[index]:
        letter_count[index] = number


letters = string.ascii_lowercase

#defining a sort function
def xCoo(arr):
    return arr[0][0]

def yCoo(arr):
    return arr[0][1]


#loading files
for file in files[:]:
    if file.endswith('JPG'):
        photos.append(cv.imread('original\\'+file))




#color detection

lower_blue = np.array([100,40,0])
upper_blue = np.array([135,255,255])



Blue_filtered=[]

    
for photo in photos:
    
    #converting colorspace:
    photo = cv.resize(photo,(1920,1080))
    hsv = cv.cvtColor(photo,cv.COLOR_BGR2HSV)

    #filtered version of Picture:

    #hsv = cv.GaussianBlur(hsv,(3,3),2)
    
    filtered = cv.inRange(hsv,lower_blue,upper_blue)
    filtered = cv.bitwise_not(filtered)
   
    
    Blue_filtered.append(filtered)

    

#contour detection:


for picture in Blue_filtered[:]:

    #picture = cv.GaussianBlur(picture,(5,5),5,1)
    #_,picture = cv.threshold(picture,100,130,cv.THRESH_BINARY)
        
    _,ctrs, hier = cv.findContours(picture, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    

    picture=cv.cvtColor(picture,cv.COLOR_GRAY2RGB)

    

    #creting an array of rectangles with areas

    rects_and_area = [[cv.boundingRect(ctr),cv.contourArea(ctr)] for ctr in ctrs]

    
    #sorting the areas and rectangles from left to right 
    rects_and_area.sort(key=xCoo)

            
            
    untouched= copy.deepcopy(picture)

   
        
        
    for rect in rects_and_area:
        if(rect[1]<upper_bound_rectangle_area and rect[1]>lower_bound_rectangle_area):
                       
            cv.rectangle(untouched, (rect[0][0],rect[0][1]) , (rect[0][0]+rect[0][2],rect[0][1]+rect[0][3]), (0, 255, 0), 3)
        """else:
            cv.rectangle(untouched, (rect[0][0],rect[0][1]) , (rect[0][0]+rect[0][2],rect[0][1]+rect[0][3]), (255, 0, 0), 3)

        """

    
    

    

    
    cv.imshow('untouched',untouched)
    cv.waitKey(0)

    

    
    for rectangle in rects_and_area:

        
            
        if(rectangle[1]<upper_bound_rectangle_area and rectangle[1]>lower_bound_rectangle_area):
            offset=0       
            if (rectangle[0][1]-offset>=0 and
                rectangle[0][1]+rectangle[0][3]+offset<=int(picture.shape[1]) and
                rectangle[0][0]-offset>=0 and
                rectangle[0][0]+rectangle[0][2]+offset<=int(picture.shape[0])):
                
                out_cropped=picture[rectangle[0][1]-offset:rectangle[0][1]+rectangle[0][3]+offset,
                                    rectangle[0][0]-offset:rectangle[0][0]+rectangle[0][2]+offset]
            else:
                out_cropped=picture[rectangle[0][1]:rectangle[0][1]+rectangle[0][3],
                                    rectangle[0][0]:rectangle[0][0]+rectangle[0][2]]
            out_cropped=cv.resize(out_cropped,(32,32))
                
            cv.imshow('cropped',out_cropped)
            letter = chr(cv.waitKey(0))
            
            if(letter!=' '):
            
                index = letters.index(letter)

                cv.imwrite('label_HighRes\\{}_{}.jpg'.format(letters[index],letter_count[index]),out_cropped)

                letter_count[index] = letter_count[index]+1


            
            
                
                
                
            

                
        

   













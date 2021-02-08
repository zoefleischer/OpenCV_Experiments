# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:41:05 2021

@author: Zoe Fleischer
"""

# Task 1:
# =======

# a) Load the text file 'mousePos.txt' into Python and extract the data of the cursor positions (x and y) in an automated manner.
# b) Visualize the current and past 20 positions with the image 'Desktop.png' as background.
# c) Export b) as a video file.
# d) Calculate the mean cursor position.
# e) Calculate and visualize a Heatmap, giving information about the time the cursor was staying in an area. The result should look somewhat like 'Heatmap_example.jpg'.

#------IMPORTS------#

import pandas as pd
import cv2
import seaborn as sns
import numpy as np
import glob
import matplotlib.pyplot as plt


#----------------------------------EXTRACT DATA----------------------------------------#
#####################################################################################

def extract_data(filepath):
    xlist=[]
    ylist=[]
    timestamp=[]
    with open(filepath, "r") as file:
        for line in file:
            #extracting only the number of the x and y coordinates
            xlist.append(int(line.split(' ')[5].replace("x:","")))
            ylist.append(int(line.split(' ')[6].replace("y:","")))
            timestamp.append(int(line.split(' ')[3].replace('.','').replace(',','')))
    xy_dict=list(zip(xlist,ylist))  #zipping coordinates into a list of tuples
    xy_df= pd.DataFrame(xy_dict)    #visualizing necessary info as data frame
    xy_df.columns=['x-position','y-position']
    xy_df['timestamp']=timestamp
    xy_df['x & y']=xy_dict
    return xy_df


#----------------------------------COUNT OCCURENCES-----------------------------#
#################################################################################

from collections import Counter
countlist=[]
def counter(coordinates):
    data = xy_df['x & y']
    elem = coordinates
    # Given list and element
    cnt = Counter(data)
    countlist.append(cnt[elem])
    return countlist
    
xy_df['x & y'].apply(counter)

xy_df['count']=countlist



#----------------------------------EXTRACT UNIQUE VALUES-----------------------------#
#####################################################################################

to_keep = []
for i in xy_df["x & y"].unique():
    to_keep.append(xy_df[xy_df["x & y"] == i].index[0])
    
uniques=[]
for i in to_keep:
    uniques.append(xy_df.iloc[i])
    
    
uniques=pd.DataFrame(uniques)
display(uniques[uniques['count']>20].sort_values(by='count'))
print(len(uniques))


#------------------------DRAW COORDINATES ON PIC-------------------------#
##########################################################################

def draw_coordinates(coordinate_list):
    i=0
    xy= list(uniques['x & y'])[0:50]  #using unique positions of mouse movement (excluding data while mouse is resting on same coordinates)
    for coordinates in xy:
        img=cv2.imread('Desktop' + str(i)+ '.jpg', 1)
        image = cv2.circle(img, coordinates, radius=10, color=(0, 0, 255), thickness=-1) #marking the given coordinates with red dot
        print(coordinates)
        image=cv2.imwrite('Desktop' + str(i+1)+ '.jpg',image)
        i += 1

    return ("Image "+str(i)+" saved!")


#-------------------EXPORT AS VIDEO FILE--------------------------------#
########################################################################

def save_video(path):
    img_array = []
    for filename in glob.glob(path +"\\*.jpg"):
        img = cv2.imread(filename)    #loading all pictures
        print(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) #turning them into list of arrays

    print(len(img_array)) 

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 4, size)

    for i in range(len(img_array)):  #writing each pic into a video
        out.write(img_array[i])
    out.release()
    return ("Video saved!")


#----------------------------------HEATMAP----------------------------------------#
###################################################################################

#---------------TRACKING MOUSE MOVEMENT--------------------#
i=0
xy= list(xy_dict)
for coordinates in xy[0:5000]:
    img=cv2.imread('blank.jpg', 1) #open a blank page
    image = cv2.circle(img, coordinates, radius=3, color=(0, 0, 255), thickness=-1) #draw a dot for each coordinate from list
    cv2.imwrite('blank.jpg',image)
    print(i) #counting the pics it saved i.e. showing progress
    i += 1

#--------------PLOTTING MOUSE MOVEMENT SCATTERPLOT------------------#
plot=sns.scatterplot(data=xy_df, x='x-position', y='y-position',alpha=0.05) # plotting all mouse coordinates with opacity to make resting visible
plt.figure(figsize=(36, 10))
plot.figure.savefig("output.png")


#--------------PLOTTING MOUSE MOVEMENT HEATMAP------------------#

uniques1=uniques[uniques['count']>10].sort_values(by='count')
table = pd.pivot_table(uniques1, values='count', index=['y-position'],columns=['x-position'])

fig = plt.figure(figsize=(36,10))
plot = sns.heatmap(table, cmap='BuPu')
plot.set_title("Heatmap of Mouse Movements")
plot.figure.savefig("mouse.png")


#------------------APPLY HEATMAP ON IMAGE----------------#

img2=cv2.imread('Desktop.png', 1)
img = cv2.imread('mouse.png', 1)
print('Done')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #turning image into black&white
print('Done')
#heatmap
heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET) #turning black&white into heatmap
print('Done')

fin = cv2.addWeighted(heatmap_img, 0.7, img2, 0.3, 0) #superimpose heatmap on original pic
print('Done')
cv2.imwrite('Heatmapp.png',fin)
print('Saved')


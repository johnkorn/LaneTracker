#! /usr/bin/env python
import os
import sys
import csv
import cv2
import glob
import numpy as np
from numpy.core.numeric import nan

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(data))
    return data[s<m]

if __name__ == "__main__":

    cv2.namedWindow('Lane Markers', flags=cv2.WINDOW_NORMAL)
    imgs = glob.glob("images/*.png")
    
    for fname in imgs:
        # Load image and prepare output image
        img_original = cv2.imread(fname)

        # cut ROI
        (height_original, width_original) = img_original.shape[:2]
        height_start = height_original*10/18
        height_end = height_original*5/6
        img = img_original[height_start:height_end,:]
        (h, w) = img.shape[:2]
        
        # smooth image and convert to grayscale
        img = cv2.GaussianBlur(img,(5,5),0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # vertical and horizontal filtering
        v_kernel = np.matrix('-1;1;1;-1')
        h_kernel = (-1,1,1,-1)
        filter_v = cv2.filter2D(img_gray, -1, v_kernel)
        filter_h = cv2.filter2D(img_gray, -1, h_kernel)
        filtered=filter_v+filter_h
        
        # Law's texture filters
        e_vert = np.matrix('-1;2;0;2;1')        
        e_hor = (-1,2,0,2,1)
        s_vert = np.matrix('-1;0;2;0;1')        
        s_hor = (-1,0,2,0,1)
        img_ee = cv2.filter2D(cv2.filter2D(255-img_gray, -1, e_hor),-1,e_vert)
        img_ss = cv2.filter2D(cv2.filter2D(255-img_gray, -1, s_hor),-1,s_vert)
        laws = 255-img_ee+255-img_ss
                
                
        # define the seed point        
        seed_point = (w*4/9,h/2);
        diff = 2;
        # define input mask as image edges
        edges = cv2.Canny(img_gray,100,100)
        mask = cv2.copyMakeBorder(edges,1,1,1,1,cv2.BORDER_REPLICATE)
        
        # flood fill (img_gray is changed!) 
        comps = cv2.floodFill(img_gray, mask, seed_point, 0, diff, diff, flags=8 | (255 << 8))    
        
        # add filtered image to previously detected edges and dilate    
        ret,edges_extended = cv2.threshold(edges+filtered,127,255,cv2.THRESH_BINARY)
        kernel = 255*np.ones((5,5),np.uint8)
        edges_dilated = cv2.dilate(edges_extended,kernel,iterations = 1) 
        
        # remove unnecessary parts from flood-filled image, add Law's filters and dilate
        ret,filled_bin = cv2.threshold(255-img_gray,254,255,cv2.THRESH_BINARY)
        filled_bin = cv2.Canny(filled_bin+laws,100,100)
        filled_bin = cv2.dilate(filled_bin,kernel,iterations = 1)
        
        # get feature image by multiplying two previously dilated images
        feature_image = 255*np.multiply(filled_bin/255,edges_dilated/255)        
         
        # Hough Transform to detect lines
        minLineLength = 200
        maxLineGap = 10
        lines = cv2.HoughLinesP(feature_image,1,np.pi/180,100,minLineLength,maxLineGap)
        
        left_x='None'
        right_x='None'
        
        # filter candidate lines
        if(lines!=None):
            left_karr = [] 
            right_karr = []
            left_xarr = []
            right_xarr = []
            for line in lines:                
                (x1,y1,x2,y2) = line[0]
                if (x2==x1 or y2==y1):
                    continue
                
                # get line equation params (y=kx+b)   
                y2 = y2 + height_start # we must work with original image now
                y1 = y1 + height_start             
                k = 1.0*(y2-y1)/(x2-x1)                
                b = y1-k*x1
                y = height_original
                x = int((y-b)/k)  #intercept candidate               
                if(x>=-0.5*width_original and x<=width_original/2):   # left intercept candidate
                    left_karr.append(k)
                    left_xarr.append(x)
                else:
                    if(x>width_original/2 and x<=1.5*width_original): # right intercept candidate
                        right_karr.append(k)
                        right_xarr.append(x)                    
            
            # get intercepts as means of candidates and draw markers
            left_k_np = reject_outliers(np.array(left_karr))
            left_x_np = reject_outliers(np.array(left_xarr))
            right_k_np = reject_outliers(np.array(right_karr))
            right_x_np = reject_outliers(np.array(right_xarr))
                        
            y_end = height_original/2
            color = (0,255,255) # yellow
            
            if(len(left_x_np)>0):
                left_x = int(np.mean(left_x_np))            
                left_k = np.mean(left_k_np)
                xl_end = int((y_end-(height_original-left_k*left_x))/left_k)
                cv2.line(img_original,(left_x,height_original),(xl_end,y_end),color)
            
            if(len(right_x_np)>0):                        
                right_x = int(np.mean(right_x_np))                
                right_k = np.mean(right_k_np)
                xr_end = int((y_end-(height_original-right_k*right_x))/right_k)
                cv2.line(img_original,(right_x,height_original),(xr_end,y_end),color)
                   
                   
        # Show image        
        cv2.imshow('Lane Markers', img_original)
        key = cv2.waitKey(50)
        if key == 27:
            sys.exit(0)
                
    cv2.destroyAllWindows();

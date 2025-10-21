import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import turtle as t
from time import sleep

def segment(image_path, K=5, draw=False):
    img = cv.imread(image_path) # Read the image
    img = cv.blur(img, (5,5), 0) # Blur to reduce noise
    Z = img.reshape((-1,3)) # Reshape the image to a 2D array of pixels
    Z = np.float32(Z) # Convert to float32

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0) # Define criteria
    K = 5  # Number of clusters
    _, labels, centers = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS) # Apply k-means

    centers = np.uint8(centers) # Convert back to uint8
    segmented_data = centers[labels.flatten()] # Map labels to center colors
    segmented_image = segmented_data.reshape(img.shape) # Reshape back to original image shape
    gray = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY) # Convert to grayscale

    #im = cv.imread('Edward.jpg')
    #assert im is not None, "file could not be read, check with os.path.exists()"
    #imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #ret, thresh = cv.threshold(gray,80,255, cv.THRESH_TOZERO_INV)¨


    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10) # Adaptive thresholding
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) # Find contours
    contours = [cnt for cnt in contours if cv.contourArea(cnt) >= 10] # Filter small contours

        

    #t = t.Turtle()
    #t.speed(0)  
    #t.penup()


    
    if(draw): # Draw contours using turtle graphics
        for cnt in contours: 
            area = cv.contourArea(cnt)
            if  area < 0 :
                continue
            points = cnt[:,0,:]
            #print(points)
            points = points[::5]  # Downsample for faster drawing
            t.penup()
            t.goto(points[0,0]-img.shape[1]/2, img.shape[0]/2-points[0,1])
            t.pendown()
            for p in points:
                t.goto(p[0]-img.shape[1]/2, img.shape[0]/2-p[1])

    return contours
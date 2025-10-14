import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import turtle as t
from time import sleep

def segment(image_path, K=5, draw=False):
    img = cv.imread(image_path)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5  # antal grupper
    _, labels, centers = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    gray = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY)

    #im = cv.imread('Edward.jpg')
    #assert im is not None, "file could not be read, check with os.path.exists()"
    #imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,40, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) >= 50]

        

    #t = t.Turtle()
    #t.speed(0)  
    #t.penup()
    if(draw):
        for cnt in contours:
            print(cnt)
            print(len(cnt))
            print(cv.contourArea(cnt))
            area = cv.contourArea(cnt)
            if  area < 50 :
                continue
            points = cnt[:,0,:]
            #print(points)
            points = points[::3]  # downsample for faster drawing
            t.penup()
            t.goto(points[0,0]-img.shape[1]/2, img.shape[0]/2-points[0,1])
            t.pendown()
            for p in points:
                t.goto(p[0]-img.shape[1]/2, img.shape[0]/2-p[1])

    return contours
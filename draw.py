import math 
import time
import cv2

image = cv2.imread('result.png')

image = cv2.line(image,(100,100),(200,200),(0,255,0),1)

cv2.imshow('output',image)
cv2.waitKey(2000)

image = cv2.line(image,(100,150),(200,200),(200,255,0),3)
cv2.imshow('output',image)
cv2.waitKey(2000)

################################################################################
### This is a small project to demonstrate the power of Image Processing,    ###
### and leveraging it to simulate an Octapad.                                ###
################################################################################

import os
import cv2
import time
import urllib
import pygame
import numpy             as     np
from   collections       import deque
from   scipy.interpolate import griddata


###Global variables
flowChartExecOnce = True
colorImage = None
grayImage = None
binaryImage = None
hsvImage = None
redMask = None
mouseX = None
mouseY = None
topLeft     = [-1,-1]
topRight    = [-1,-1]
bottomLeft  = [-1,-1]
bottomRight = [-1,-1]


def drawGrid():
    cv2.line(colorImage, (topLeft[0],topLeft[1]), (topRight[0],topRight[1]), (0,255,0), 1)
    cv2.line(colorImage, (topLeft[0],topLeft[1]), (bottomLeft[0],bottomLeft[1]), (0,255,0), 1)
    cv2.line(colorImage, (topRight[0],topRight[1]), (bottomRight[0],bottomRight[1]), (0,255,0), 1)
    cv2.line(colorImage, (bottomLeft[0],bottomLeft[1]), (bottomRight[0],bottomRight[1]), (0,255,0), 1)
    cv2.line(colorImage, ((topLeft[0]+topRight[0])/2,(topLeft[1]+topRight[1])/2), ((bottomLeft[0]+bottomRight[0])/2,(bottomLeft[1]+bottomRight[1])/2), (0,255,0), 1)
    cv2.line(colorImage, ((topLeft[0]+bottomLeft[0])/2,(topLeft[1]+bottomLeft[1])/2), ((topRight[0]+bottomRight[0])/2,(topRight[1]+bottomRight[1])/2), (0,255,0), 1)
    cv2.line(colorImage, ((topLeft[0]*3+topRight[0])/4,(topLeft[1]*3+topRight[1])/4), ((bottomLeft[0]*3+bottomRight[0])/4,(bottomLeft[1]*3+bottomRight[1])/4), (0,255,0), 1)
    cv2.line(colorImage, ((topLeft[0]+topRight[0]*3)/4,(topLeft[1]+topRight[1]*3)/4), ((bottomLeft[0]+bottomRight[0]*3)/4,(bottomLeft[1]+bottomRight[1]*3)/4), (0,255,0), 1)
    cv2.line(colorImage, ((topLeft[0]*3+bottomLeft[0])/4,(topLeft[1]*3+bottomLeft[1])/4), ((topRight[0]*3+bottomRight[0])/4,(topRight[1]*3+bottomRight[1])/4), (0,0,255), 1)
    cv2.line(colorImage, ((topLeft[0]+bottomLeft[0]*3)/4,(topLeft[1]+bottomLeft[1]*3)/4), ((topRight[0]+bottomRight[0]*3)/4,(topRight[1]+bottomRight[1]*3)/4), (0,0,255), 1)
    

def draw_circle(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y


def locateOctaPad():
    global mouseX, mouseY, topLeft, topRight, bottomLeft, bottomRight
    cv2.namedWindow('colorImage')
    cv2.setMouseCallback('colorImage',draw_circle)
    hsvImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
    coordinatesLocated = False
    while not(coordinatesLocated):
        cv2.imshow("colorImage", colorImage)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('i'):
            topLeft = [mouseX,mouseY]
            print topLeft
            print hsvImage[mouseY, mouseX]
        elif k == ord('o'):
            topRight = [mouseX,mouseY]
            print topRight
        elif k == ord('k'):
            bottomLeft = [mouseX,mouseY]
            print bottomLeft
        elif k == ord('l'):
            bottomRight = [mouseX,mouseY]
            print bottomRight
        elif k == ord('q'):
            coordinatesLocated = True
    # topLeft, topRight, bottomLeft, bottomRight = [35, 39], [148, 32], [13, 108], [165, 104]


def imgProcess():
    # Convert BGR to HSV
    global colorImage, hsvImage
    hsvImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170, 50, 150])
    upper_red = np.array([190, 170, 255])
    redMask = cv2.inRange(hsvImage, lower_red, upper_red)
    # res = cv2.bitwise_and(colorImage,colorImage, mask= redMask)
    cv2.imshow('redMask',redMask)
    # cv2.imshow('res',res)


def flowChart():
    global colorImage
    global flowChartExecOnce
    if flowChartExecOnce:
        locateOctaPad()
        print topLeft, topRight, bottomLeft, bottomRight
        cv2.destroyAllWindows()
        drawGrid()
        cv2.imshow("colorImage", colorImage)
        flowChartExecOnce = False
    else:
        imgProcess()


def main():
    pygame.init()
    global colorImage
    imageORvideo = True #True==image
    if imageORvideo:
        print "Working on image..."
        global colorImage
        colorImage = cv2.imread('Temp_Storage/frameTOP.jpg')
        flowChart()
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)
    else:
        print "Working on video..."
        IP_addr_TOP  = '192.168.1.2:8080'
        ### Obtain the frames for processing
        # top_frame, side_frame  = streamVideo(IP_addr_TOP, IP_addr_SIDE)
        streamTOP  = urllib.urlopen('http://'+IP_addr_TOP+'/video?.mjpeg')
        bytesTOP   = ''
        displayOutput = True
        fixImage = False
        print "Press 'q' to exit"
        while True:
            bytesTOP  += streamTOP.read(1024)
            aTOP  = bytesTOP.find('\xff\xd8')
            bTOP  = bytesTOP.find('\xff\xd9')
            if aTOP!=-1 and bTOP!=-1:
                jpgTOP    = bytesTOP [aTOP:bTOP+2]
                bytesTOP  = bytesTOP [bTOP+2:]
                colorImage  = cv2.imdecode(np.fromstring(jpgTOP, dtype=np.uint8),cv2.IMREAD_COLOR)
                if not(fixImage):
                    cv2.imshow('Mobile IP CameraTOP',colorImage)
                    if cv2.waitKey(1) & 0xFF == ord('1'):
                        cv2.destroyAllWindows()
                        flowChart()
                        fixImage = True
                else:
                    flowChart()
                    if displayOutput:
                        cv2.imshow('Mobile IP CameraTOP',colorImage)
            # # Exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)


if __name__ == "__main__":
    main()     
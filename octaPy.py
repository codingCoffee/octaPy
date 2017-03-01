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
center = None
gridCtr1 = gridCtr2 = gridCtr3 = gridCtr4 = gridCtr5 = gridCtr6 = gridCtr7 = gridCtr8 = None



def maskGrid():
    global colorImage, GRID, gridCtr1, gridCtr2, gridCtr3, gridCtr4, gridCtr5, gridCtr6, gridCtr7, gridCtr8
    cv2.line(colorImage, (topLeft[0],topLeft[1]), (topRight[0],topRight[1]), (0,255,0), 1)
    cv2.line(colorImage, (topLeft[0],topLeft[1]), (bottomLeft[0],bottomLeft[1]), (0,255,0), 1)
    cv2.line(colorImage, (topRight[0],topRight[1]), (bottomRight[0],bottomRight[1]), (0,255,0), 1)
    cv2.line(colorImage, (bottomLeft[0],bottomLeft[1]), (bottomRight[0],bottomRight[1]), (0,255,0), 1)
    
    cv2.line(colorImage, ((topLeft[0]+bottomLeft[0])/2,(topLeft[1]+bottomLeft[1])/2), ((topRight[0]+bottomRight[0])/2,(topRight[1]+bottomRight[1])/2), (0,255,0), 1) #center horizonatal
    cv2.line(colorImage, ((topLeft[0]+topRight[0])/2,(topLeft[1]+topRight[1])/2), ((bottomLeft[0]+bottomRight[0])/2,(bottomLeft[1]+bottomRight[1])/2), (0,255,0), 1) #center vertical
    cv2.line(colorImage, ((topLeft[0]*3+topRight[0])/4,(topLeft[1]*3+topRight[1])/4), ((bottomLeft[0]*3+bottomRight[0])/4,(bottomLeft[1]*3+bottomRight[1])/4), (0,255,0), 1) #left vertical
    cv2.line(colorImage, ((topLeft[0]+topRight[0]*3)/4,(topLeft[1]+topRight[1]*3)/4), ((bottomLeft[0]+bottomRight[0]*3)/4,(bottomLeft[1]+bottomRight[1]*3)/4), (0,255,0), 1) #right verticals
    
    cv2.line(colorImage, ((topLeft[0]*3+bottomLeft[0])/4,(topLeft[1]*3+bottomLeft[1])/4), ((topRight[0]*3+bottomRight[0])/4,(topRight[1]*3+bottomRight[1])/4), (0,0,255), 1)
    cv2.line(colorImage, ((topLeft[0]+bottomLeft[0]*3)/4,(topLeft[1]+bottomLeft[1]*3)/4), ((topRight[0]+bottomRight[0]*3)/4,(topRight[1]+bottomRight[1]*3)/4), (0,0,255), 1)
    


def playMusic():
    global center, gridCtr1, gridCtr2, gridCtr3, gridCtr4, gridCtr5, gridCtr6, gridCtr7, gridCtr8
    if (cv2.pointPolygonTest(gridCtr1, center, False)>0):
        pygame.mixer.music.load('cowbell9.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr2, center, False)>0):
        pygame.mixer.music.load('hihat22.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr3, center, False)>0):
        pygame.mixer.music.load('tomtomdrum6.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr4, center, False)>0):
        pygame.mixer.music.load('tomtomdrum7.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr5, center, False)>0):
        pygame.mixer.music.load('cowbell9.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr6, center, False)>0):
        pygame.mixer.music.load('hihat22.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr7, center, False)>0):
        pygame.mixer.music.load('tomtomdrum6.wav')
        pygame.mixer.music.play(1)
    elif (cv2.pointPolygonTest(gridCtr8, center, False)>0):
        pygame.mixer.music.load('tomtomdrum7.wav')
        pygame.mixer.music.play(1)
    else:
        pygame.mixer.music.stop()
    center = None


def imgProcess():
    # Convert BGR to HSV
    global colorImage, hsvImage, redMask, center
    hsvImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
    lower_red = np.array([10, 150, 140])
    upper_red = np.array([30, 220, 210])
    redMask = cv2.inRange(hsvImage, lower_red, upper_red)
    redMask = cv2.erode(redMask, None, iterations=2)
    redMask = cv2.dilate(redMask, None, iterations=2)
    # res = cv2.bitwise_and(colorImage,colorImage, mask= redMask)
    maskGrid()
    cv2.imshow('redMask',redMask)
    # cv2.imshow('Edited colorImage',colorImage)
    # cv2.imshow('res',res)

    cnts = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    # center = None
 
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # print center[0], center[1]
        # only proceed if the radius meets a minimum size
        if radius > 2:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            # cv2.circle(colorImage, (int(x), int(y)), int(radius),
            #     (0, 255, 255), 2)
            cv2.circle(colorImage, center, 2, (255, 0, 0), -1)
 
    # # update the points queue
    # pts.appendleft(center)

    # # loop over the set of tracked points
    # for i in xrange(1, len(pts)):
    #     # if either of the tracked points are None, ignore
    #     # them
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
 
    #     # otherwise, compute the thickness of the line and
    #     # draw the connecting lines
    #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
    # show the frame to our screen
    # cv2.imshow("colorImage", colorImage)
    playMusic()


def drawGrid():
    global GRID, gridCtr1, gridCtr2, gridCtr3, gridCtr4, gridCtr5, gridCtr6, gridCtr7, gridCtr8
    # cv2.line(colorImage, (topLeft[0],topLeft[1]), (topRight[0],topRight[1]), (0,255,0), 1)
    # cv2.line(colorImage, (topLeft[0],topLeft[1]), (bottomLeft[0],bottomLeft[1]), (0,255,0), 1)
    # cv2.line(colorImage, (topRight[0],topRight[1]), (bottomRight[0],bottomRight[1]), (0,255,0), 1)
    # cv2.line(colorImage, (bottomLeft[0],bottomLeft[1]), (bottomRight[0],bottomRight[1]), (0,255,0), 1)
    
    # cv2.line(colorImage, ((topLeft[0]+bottomLeft[0])/2,(topLeft[1]+bottomLeft[1])/2), ((topRight[0]+bottomRight[0])/2,(topRight[1]+bottomRight[1])/2), (0,255,0), 1) #center horizonatal
    # cv2.line(colorImage, ((topLeft[0]+topRight[0])/2,(topLeft[1]+topRight[1])/2), ((bottomLeft[0]+bottomRight[0])/2,(bottomLeft[1]+bottomRight[1])/2), (0,255,0), 1) #center vertical
    # cv2.line(colorImage, ((topLeft[0]*3+topRight[0])/4,(topLeft[1]*3+topRight[1])/4), ((bottomLeft[0]*3+bottomRight[0])/4,(bottomLeft[1]*3+bottomRight[1])/4), (0,255,0), 1) #left vertical
    # cv2.line(colorImage, ((topLeft[0]+topRight[0]*3)/4,(topLeft[1]+topRight[1]*3)/4), ((bottomLeft[0]+bottomRight[0]*3)/4,(bottomLeft[1]+bottomRight[1]*3)/4), (0,255,0), 1) #right verticals
    
    # cv2.line(colorImage, ((topLeft[0]*3+bottomLeft[0])/4,(topLeft[1]*3+bottomLeft[1])/4), ((topRight[0]*3+bottomRight[0])/4,(topRight[1]*3+bottomRight[1])/4), (0,0,255), 1)
    # cv2.line(colorImage, ((topLeft[0]+bottomLeft[0]*3)/4,(topLeft[1]+bottomLeft[1]*3)/4), ((topRight[0]+bottomRight[0]*3)/4,(topRight[1]+bottomRight[1]*3)/4), (0,0,255), 1)
    
    GRID = [[[topLeft[0],topLeft[1]],                                   [(topLeft[0]*3+topRight[0])/4,(topLeft[1]*3+topRight[1])/4],                                                                                [(topLeft[0]+topRight[0])/2,(topLeft[1]+topRight[1])/2],                                                                            [(topLeft[0]+topRight[0]*3)/4,(topLeft[1]+topRight[1]*3)/4],                                                                                [topRight[0],topRight[1]]],
        [[(topLeft[0]+bottomLeft[0])/2,(topLeft[1]+bottomLeft[1])/2],   [((topLeft[0]*3+topRight[0])/4+(bottomLeft[0]*3+bottomRight[0])/4)/2, ((topLeft[1]*3+topRight[1])/4+(bottomLeft[1]*3+bottomRight[1])/4)/2], [((topLeft[0]+topRight[0])/2+(bottomLeft[0]+bottomRight[0])/2)/2, ((topLeft[1]+topRight[1])/2+(bottomLeft[1]+bottomRight[1])/2)/2], [((topLeft[0]+topRight[0]*3)/4+(bottomLeft[0]+bottomRight[0]*3)/4)/2,((topLeft[1]+topRight[1]*3)/4+(bottomLeft[1]+bottomRight[1]*3)/4)/2],  [(topRight[0]+bottomRight[0])/2,(topRight[1]+bottomRight[1])/2]],
        [[bottomLeft[0],bottomLeft[1]],                                 [(bottomLeft[0]*3+bottomRight[0])/4,(bottomLeft[1]*3+bottomRight[1])/4],                                                                    [(bottomLeft[0]+bottomRight[0])/2,(bottomLeft[1]+bottomRight[1])/2],                                                                [(bottomLeft[0]+bottomRight[0]*3)/4,(bottomLeft[1]+bottomRight[1]*3)/4],                                                                    [bottomRight[0],bottomRight[1]]]]
    gridCtr1 = np.array([GRID[0][0],GRID[0][1],GRID[1][1],GRID[1][0]], dtype=np.int32)
    gridCtr2 = np.array([GRID[0][1],GRID[0][2],GRID[1][2],GRID[1][1]], dtype=np.int32)
    gridCtr3 = np.array([GRID[0][2],GRID[0][3],GRID[1][3],GRID[1][2]], dtype=np.int32)
    gridCtr4 = np.array([GRID[0][3],GRID[0][4],GRID[1][4],GRID[1][3]], dtype=np.int32)
    gridCtr5 = np.array([GRID[1][0],GRID[1][1],GRID[2][1],GRID[2][0]], dtype=np.int32)
    gridCtr6 = np.array([GRID[1][1],GRID[1][2],GRID[2][2],GRID[2][1]], dtype=np.int32)
    gridCtr7 = np.array([GRID[1][2],GRID[1][3],GRID[2][3],GRID[2][2]], dtype=np.int32)
    gridCtr8 = np.array([GRID[1][3],GRID[1][4],GRID[2][4],GRID[2][3]], dtype=np.int32)


def draw_circle(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y


def locateOctaPad():
    global mouseX, mouseY, topLeft, topRight, bottomLeft, bottomRight
    cv2.namedWindow('locOctPad')
    cv2.setMouseCallback('locOctPad',draw_circle)
    hsvImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
    coordinatesLocated = False
    while not(coordinatesLocated):
        cv2.imshow("locOctPad", colorImage)
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
    imageORvideo = False #True==image
    if imageORvideo:
        print "Working on image..."
        global colorImage
        colorImage = cv2.imread('Temp_Storage/frameTOP.jpg')
        flowChart()
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)
    else:
        print "Working on video..."
        IP_addr_TOP  = '10.42.0.197:8080'
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
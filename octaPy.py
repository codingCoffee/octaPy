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
findSquare_endTime     = 0
Approx_Square_Countour = 0

def findSquare(Modified_Octapad_Image):
    global Approx_Square_Countour
    Temp_Image = Modified_Octapad_Image.copy()
    #cv2.imshow('Temp_Image', Temp_Image)
    ###Create a copy of the Modified_Octapad_Image to implement cv2.findContours() on,
    ## since after applying this method the image gets distorted for some reason.
    ###We are using the .copy() method to create the image since using something like
    ## img1 = img2, simply creates an object pointing to the original one. So altering
    ## either of the images also alters the other image and hence using it makes no sense
    tempo_, Contours, Hierarchy = cv2.findContours(Temp_Image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #OGimg = cv2.cvtColor(Temp_Image,cv2.COLOR_GRAY2RGB)
    #cv2.drawContours(OGimg,Contours,-1,(0,255,0),1)
    #cv2.imshow("Modified_Octapad_Image.png", OGimg)
    ###Find the contours in the image
    ###cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
    ###(image) ~input binary image
    ###Refer the link below for more info
    ##http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
    Required_Square_Contour = None
    Required_Contour_Area = 0
    for Contour in Contours:
        Contour_Area = cv2.contourArea(Contour)
        ###Calculates the area enclosed by the vector of 2D points denoted by 
        ## the variable Contour
        if Contour_Area > 500:
            if Contour_Area > Required_Contour_Area:
                Required_Contour_Area = Contour_Area
                Required_Square_Contour = Contour
    ###Code for finding out the largest contour (on the basis of area)
    Perimeter_of_Contour = cv2.arcLength(Required_Square_Contour, True)
    ###Calculates a contour perimeter or a curve length
    ###cv2.arcLength(curve, closed)
    ###(curve) ~Input vector of 2D points
    ###(closed) ~Flag indicating whether the curve is closed or not
    Temp_Square_Countour = cv2.approxPolyDP(Required_Square_Contour, 0.05*Perimeter_of_Contour, True)
    ###Approximates a polygonal curve(s) with the specified precision
    ###cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])
    Temp_Square_Countour = Temp_Square_Countour.tolist()
    Approx_Square_Countour = []
    for Temp_Var_1 in Temp_Square_Countour:
        for Temp_Var_2 in Temp_Var_1:
            Temp_Var_2[0], Temp_Var_2[1] = Temp_Var_2[1], Temp_Var_2[0]
            Approx_Square_Countour.append(Temp_Var_2)
    ###Temp_Square_Countour has the coordinates inside a list within a list, 
    ## hence to extract it we're doing this. Also we're changing (row, column) i.e.
    ## (y, x) to (column, row) i.e. (x, y)
    ###This was done because the griddata function from the scipy library 
    ## takes in values as (column, row) i.e. (x,y) instead of (row, column) i.e (y,x)
    Approx_Square_Countour = deque(Approx_Square_Countour)
    ###Applying deque function on anything converts it into a queue and we can use
    ## functions like popleft() etc on it, as if it were a queue 
    Min_Sum = 9999999
    ###Initialized to a fairly large number as we want minimum
    Counter = -1
    ###Used as counter to keep tract of the iteration number so that the
    ## location of top-left coordinate can be stored in the variable Loc
    Loc = 0
    for i in Approx_Square_Countour:
        Counter+=1
        if Min_Sum > sum(i):
            Min_Sum = sum(i)
            Loc = Counter
    if Loc != 0:
        for i in range(0,Loc):
             Approx_Square_Countour.append(Approx_Square_Countour[0])
             Approx_Square_Countour.popleft()
    ###If the sum of the x and y coordinates is minimum it would automatically
    ## mean that the coordinate refers to the top-left point of the square.
    ###We know the coordinates of the square are stored in a cyclic fashion,
    ## hence if we know the location of the top-left coordinate then we can
    ## re-arrage it by appending the 1st coordinate and then poping it.
    ## Example: (4,1,2,3)
    ## Now appending 1st loc we get (4,1,2,3,4)
    ## Now popping 1st loc we get (1,2,3,4) which is the required result
    ## That is what this code does to rearrange the coordinates
    Approx_Square_Countour[1], Approx_Square_Countour[3] = Approx_Square_Countour[3], Approx_Square_Countour[1]
    ###Flipping the location of 1st and 3rd coordinates makes the coordinate 
    ## pointer go counter-clockwise. We do this because opencv stores the 
    ## coordinate values in a clockwise fashion, however griddata function from 
    ## scipy library requires it to be in a counter-clockwise fashion
    #cv2.drawContours(Modified_Octapad_Image,[Approx_Square_Countour],0,255,10)
    Mask = np.zeros((Modified_Octapad_Image.shape),np.uint8)
    ###Creates a black image of the same size as the input image
    cv2.drawContours(Mask,[Required_Square_Contour],0,255,-1)
    cv2.drawContours(Mask,[Required_Square_Contour],0,0,2)
    ###Overwrites the black image with the area of the octapad in white
    Modified_Octapad_Image = cv2.bitwise_and(Modified_Octapad_Image,Mask)
    ###Compares the Modified_Octapad_Image and the Mask and blackens all parts 
    ## of the image other than the octapad
    #cv2.imshow('Modified_Octapad_Image', Modified_Octapad_Image)
    #print Approx_Square_Countour
    return Modified_Octapad_Image, Approx_Square_Countour


def imgPreProcess(View):
    Modified_Top_Image = cv2.cvtColor(View,cv2.COLOR_BGR2GRAY)
    # Modified_Side_Image = cv2.cvtColor(Side_view,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Modified_Octapad_Image.png", Modified_Octapad_Image)
    ###We convert the image to B&W format to easily be able to extract info
    Modified_Top_Image = cv2.GaussianBlur(Modified_Top_Image,(5,5),0)
    # Modified_Side_Image = cv2.GaussianBlur(Modified_Side_Image,(5,5),0)
    #cv2.imshow("Modified_Octapad_Image.png", Modified_Octapad_Image)
    ###Gaussian Blur is applied to remove any noise from the image
    ###https://www.youtube.com/watch?v=C_zFhWdM4ic
    ###cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    ###(src) ~input image
    ###(ksize) ~kernal size
    ###(sigmaX, sigmaY) ~.indicate the standard deviation in the x and y directions,
    ## .making both of them 0 means the gaussian kernal is automatically calculated
    Modified_Top_Image = cv2.adaptiveThreshold(Modified_Top_Image,255,1,1,19,5)
    # Modified_Side_Image = cv2.adaptiveThreshold(Modified_Side_Image,255,1,1,19,5)
    #cv2.imshow("Modified_Octapad_Image.png", Modified_Octapad_Image)
    ###Adaptive Thresholding is done to adjust for different lighting conditions
    ###cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
    ###(src) ~input image
    ###(maxValue) ~value assigned to pixels for which the contdition is satisfied
    ###(adaptiveMehod) ~ADAPTIVE_THRESH_MEAN_C == 0; ADAPTIVE_THRESH_GAUSSIAN_C == 1
    ###(thresholdType) ~THRESH_BINARY == 0; THRESH_BINARY_INV == 1
    ###(blockSize) ~.something like the kernal size
    ###(C) ~Constant subtracted from the mean or weighted mean ~.to clear the noise
    #cv2.imshow('Modified_Octapad_Image', Modified_Octapad_Image)
    return Modified_Top_Image

def playRandom(Modified_Image):
    row, column = Modified_Image.shape

    roi1 = Modified_Image[0     : 0+row/2    , 0          : column/4  ]
    roi2 = Modified_Image[0     : 0+row/2    , column/4   : column/2  ]
    roi3 = Modified_Image[0     : 0+row/2    , column/2   : 3*column/4]
    roi4 = Modified_Image[0     : 0+row/2    , 3*column/4 : column-1  ]
    roi5 = Modified_Image[row/2 : row/2+row-1, 0          : column/4  ]
    roi6 = Modified_Image[row/2 : row/2+row-1, column/4   : column/2  ]
    roi7 = Modified_Image[row/2 : row/2+row-1, column/2   : 3*column/4]
    roi8 = Modified_Image[row/2 : row/2+row-1, 3*column/4 : column-1  ]

    hist1 = cv2.calcHist([roi1],[0],None,[2],[0,256])
    hist2 = cv2.calcHist([roi2],[0],None,[2],[0,256])
    hist3 = cv2.calcHist([roi3],[0],None,[2],[0,256])
    hist4 = cv2.calcHist([roi4],[0],None,[2],[0,256])
    hist5 = cv2.calcHist([roi5],[0],None,[2],[0,256])
    hist6 = cv2.calcHist([roi6],[0],None,[2],[0,256])
    hist7 = cv2.calcHist([roi7],[0],None,[2],[0,256])
    hist8 = cv2.calcHist([roi8],[0],None,[2],[0,256])
    # cv2.imshow('roi1',roi1) #Top Left
    # cv2.imshow('roi2',roi2) #Bottom Right
    # cv2.imshow('roi3',roi3) #Top Right
    # cv2.imshow('roi4',roi4) #Bottom Left
    # print hist1, hist4
    if (hist1[1] > 600):
        pygame.mixer.music.load('cowbell9.wav')
        pygame.mixer.music.play(1)
    elif (hist2[1] > 600):
        pygame.mixer.music.load('hihat22.wav')
        pygame.mixer.music.play(1)
    elif (hist3[1] > 600):
        pygame.mixer.music.load('tomtomdrum6.wav')
        pygame.mixer.music.play(1)
    elif (hist4[1] > 600):
        pygame.mixer.music.load('tomtomdrum7.wav')
        pygame.mixer.music.play(1)
    elif (hist5[1] > 600):
        pygame.mixer.music.load('cowbell9.wav')
        pygame.mixer.music.play(1)
    elif (hist6[1] > 600):
        pygame.mixer.music.load('hihat22.wav')
        pygame.mixer.music.play(1)
    elif (hist7[1] > 600):
        pygame.mixer.music.load('tomtomdrum6.wav')
        pygame.mixer.music.play(1)
    elif (hist8[1] > 600):
        pygame.mixer.music.load('tomtomdrum7.wav')
        pygame.mixer.music.play(1)



def flowChart(View):
    global findSquare_endTime
    global Approx_Square_Countour
    ###Convert the color image to a binary image
    Modified_Image = imgPreProcess(View)
    ###To display the images, for feedback. Coment when not required
    # cv2.imshow('Modified_Top_Image' ,Modified_Top_Image)
    # cv2.imshow('Modified_Side_Image',Modified_Side_Image)
    ###Execute this at intervals of 5 seconds
    # if 5<=time.time() - findSquare_endTime:
        ###Locate the corners of the octapad
        # Approx_Square_Countour = findSquare(Modified_Top_Image)
        # findSquare_endTime = time.time()
    #Modified_Octapad_Image = findSquare(stretchSquare)
    ###Removes noise and converts the image into binary form i.e. 0's and 1's
    '''
    global glob
    if glob:
        Modified_Octapad_Image, Square_Contour = findSquare(Modified_Octapad_Image)
        glob=False
        np.savetxt('Samples.data',Square_Contour)
    '''
    ###Locates the octapad square and removes all the other stuff around it
    #Warped_Octapad_Image = stretchSquare(Modified_Octapad_Image, Square_Contour)
    ###Resizes the octapad square to a new image of 450x450 pixels
    playRandom(Modified_Image)
    return Modified_Image


def main():
    # IP addresses of both the streams
    pygame.init()
    print "Working..."
    IP_addr_TOP  = '10.42.0.197:8080'
    IP_addr_SIDE = '10.42.0.197:8080'
    ### Obtain the frames for processing
    # top_frame, side_frame  = streamVideo(IP_addr_TOP, IP_addr_SIDE)
    streamTOP  = urllib.urlopen('http://'+IP_addr_TOP+'/video?.mjpeg')
    streamSIDE = urllib.urlopen('http://'+IP_addr_SIDE+'/video?.mjpeg')
    bytesTOP   = ''
    bytesSIDE  = ''
    topORside  = True
    displayOutput = True
    print "Press 'q' to exit"
    while True:
        if topORside:
            bytesTOP  += streamTOP.read(1024)
            aTOP  = bytesTOP.find('\xff\xd8')
            bTOP  = bytesTOP.find('\xff\xd9')
            if aTOP!=-1 and bTOP!=-1:
                jpgTOP    = bytesTOP [aTOP:bTOP+2]
                bytesTOP  = bytesTOP [bTOP+2:]
                frameTOP  = cv2.imdecode(np.fromstring(jpgTOP, dtype=np.uint8),cv2.IMREAD_COLOR)
                dispTOP = flowChart(frameTOP)
                if displayOutput:
                    cv2.imshow('Mobile IP CameraTOP',dispTOP)
        else:
            bytesSIDE += streamSIDE.read(1024)
            # 0xff 0xd8 is the starting of the jpeg frame
            aSIDE = bytesSIDE.find('\xff\xd8')
            # 0xff 0xd9 is the end of the jpeg frame
            bSIDE = bytesSIDE.find('\xff\xd9')
            # Taking the jpeg image as byte stream
            if aSIDE!=-1 and bSIDE!=-1:
                jpgSIDE   = bytesSIDE[aSIDE:bSIDE+2]
                bytesSIDE = bytesSIDE[bSIDE+2:]
                # Decoding the byte stream to cv2 readable matrix format
                frameSIDE = cv2.imdecode(np.fromstring(jpgSIDE, dtype=np.uint8),cv2.IMREAD_COLOR)
                # Main code
                dispSIDE = flowChart(frameSIDE)
                # # Display
                if displayOutput:
                    cv2.imshow('Mobile IP CameraTOP',dispSIDE)
                # #cv2.imwrite("octa_image", disp)
        # topORside = not(topORside)
        # # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            displayOutput = False


if __name__ == "__main__":
    main()
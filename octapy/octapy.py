################################################################################
### This is a small project to demonstrate the power of Image Processing,    ###
### and leveraging it to simulate an Octapad.                                ###
################################################################################

import os
import time
import urllib
from collections import deque

import cv2
import numpy as np
from scipy.interpolate import griddata

from octapy.config import octaPyConfig


class octaPy(object):
    # IP addresses of both the streams
    IP_addr_TOP  = '192.168.1.3:8080'
    IP_addr_SIDE = '192.168.1.3:8080'
    FIND_SQUARE_END_TIME     = 0
    APPROX_SQUARE_COUNTOUR   = 0

    def __init__(self):
        self.IP_addr_TOP = octaPy.IP_addr_TOP
        self.IP_addr_SIDE = octaPy.IP_addr_SIDE


    def flowChart(Top_view, Side_view):
        # global FIND_SQUARE_END_TIME
        # global APPROX_SQUARE_COUNTOUR
        ###Convert the color image to a binary image
        # Modified_Top_Image, Modified_Side_Image = imgPreProcess(Top_view, Side_view)
        ###To display the images, for feedback. Coment when not required
        cv2.imshow('Modified_Top_Image' ,Top_view)
        cv2.imshow('Modified_Side_Image',Side_view)
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
        # return Approx_Square_Countour
        return True


    def streamVideo(IP_addr):
        stream = urllib.urlopen('http://'+IP_addr+'/video?.mjpeg')
        bytes=''
        # while True:
        bytes += stream.read(1024)
        # 0xff 0xd8 is the starting of the jpeg frame
        a = bytes.find('\xff\xd8')
        # 0xff 0xd9 is the end of the jpeg frame
        b = bytes.find('\xff\xd9')
        # Taking the jpeg image as byte stream
        if a!=-1 and b!=-1:
            os.system ( 'clear' )
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            # Decoding the byte stream to cv2 readable matrix format
            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
            return frame
            # Main code
            # disp = flowChart(frame)
            # # Display6
            # cv2.imshow('Mobile IP Camera',disp)
            # #cv2.imwrite("octa_image", disp)
            # print "Press 'q' to exit"
            # # Exit key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     exit(0)


    def run():
        while True:
            # Obtain the frames for processing
            top_frame  = streamVideo(octaPy.IP_addr_TOP)
            side_frame = streamVideo(octaPy.IP_addr_SIDE)
            Approx_Square_Countour = flowChart(top_frame, side_frame)

            print "Press 'q' to exit"
            # Exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)
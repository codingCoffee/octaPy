################################################################################
### This is a small project to demonstrate the power of Image Processing,    ###
### and leveraging it to simulate an Octapad.                                ###
################################################################################

def main():
    # IP addresses of both the streams
    IP_addr_TOP  = '192.168.43.206:8080'
    IP_addr_SIDE = '192.168.43.206:8080'
    while True:
        # Obtain the frames for processing
        top_frame  = streamVideo(IP_addr_TOP)
        side_frame = streamVideo(IP_addr_SIDE)
        Approx_Square_Countour = flowChart(top_frame, side_frame)

        print "Press 'q' to exit"
        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

if __name__ == "__main__":
    main()
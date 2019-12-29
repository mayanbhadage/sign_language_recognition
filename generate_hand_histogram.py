#This program contains the method to genetate hand histogram
import cv2,pickle,sys
import numpy as np
import terminal_colors as colors

# This function generates squares and the corrosponding region of image used
# built the histogram from. Make sure you covers all the squares perfectly to
# get the correct histogram.

# If you want to increse/ decrease the square count change the row / column variable
# accordingly

def genetate_squares(frame,x,y,w,h,row,col):

    image = None
    crop = None
    for i in range(row):
        for j in range(col):
            if np.any(image == None):
                image = frame[y:y+h,x:x+w]
            else:
                image = np.hstack((image,frame[y:y+h,x:x+w]))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            x += w+10
        if np.any(crop == None):
            crop = image
        else:
            crop = np.hstack((crop,image))
        image = None
        x = 920
        y += h+10
    return crop



# This function calculates histogram from the region of image (crop) we calulated in
# generate squares
def calculate_histogram(image):
     hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

     src_image = [hsv]      # source image
     channels = [0,1]       # index of channel for which we calculate histogram
     mask = None            # to find histogram of full image, it is given as “None”.
     hist_size = [180,256]  # this represents our BIN count
     ranges = [0,180,0,256] # this is our range : 0 to 180 for H and 0 256 for S we drop V

     #hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
     hist = cv2.calcHist(src_image,channels,mask,hist_size,ranges)

     # we normalize our histogram
     cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
     return hist


# This functions uses our histogram generated and and backproject
# it to detect the skin color

def back_project(hist,hsv_frame):
    dst = cv2.calcBackProject([hsv_frame], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    #applying blur
    blur = cv2.GaussianBlur(dst, (15,15), 0)
    blur = cv2.medianBlur(blur, 15)

    #generate threshold of the image
    ret, thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = cv2.merge((thresh,thresh,thresh))
    # image manupulation for better results of histogram
    eroded = cv2.erode(thresh,None,iterations=3)
    dilated = cv2.dilate(eroded, None,iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,None)
    cv2.imshow("Thresh", closed)



def get_hand_hist():
    #default webcam
    cap = cv2.VideoCapture(0)
    save_flag = False
    hand_detected = False
    x,y,w,h = 920,240,10,10
    row,col = 12,6

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error while reading from camera")
            break
        #flip the frame
        frame = cv2.flip(frame,1)

        #Resize the frame
        cv2.resize(frame,(640,480))

        #convert image color to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        image = genetate_squares(frame,x,y,w,h,row,col)
        cv2.putText(frame, "Place you hands over the green boxes.",
                 (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press c key on keyboard to capture histogram.",
                 (5, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (188,104,20), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press s key on keyboard to save the histogram.",
                 (5, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (188,104,20), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press q key on keyboard to quit!",
                 (5, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (188,104,20), 1, cv2.LINE_AA)
        cv2.imshow('frame',frame)


        key = cv2.waitKey(10)
        if key == ord('c'):
            hist = calculate_histogram(image)
            back_project(hist,hsv_frame)
            hand_detected = True


        if key == ord('s'):
            if hand_detected:
                cap.release()
                cv2.destroyAllWindows()
                # Our histogram will be saved as hand_histogram.pickle file
                with open ('hand_histogram.pickle',"wb") as file:
                    pickle.dump(hist,file)
                print(f"{colors.CBOLD}Histogram genetated.{colors.CEND}")
                break
            else:
                print(f"{colors.CYELLOW}# WARNING:{colors.CEND}Saved without histogram")
                cap.release()
                cv2.destroyAllWindows()
                break


        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


get_hand_hist()

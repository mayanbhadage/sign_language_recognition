import cv2
from terminal_colors import Colors as color
import numpy as np
import sys, os ,pickle,random

IMAGES_PATH = "images"
IMAGES_CLASS_PATH = ""

OVERWRITTEN = False

# This funtion validate the input data
def validate_input():
    try:
        label_name = sys.argv[1]            #First command line argument
        num_samples = int(sys.argv[2])      #Second command line argument
        global IMAGES_CLASS_PATH
        IMAGES_CLASS_PATH = os.path.join(IMAGES_PATH,label_name)
        return (label_name,num_samples)
    except:
        print(f"{color.CRED}ERROR:{color.CEND} Arguments Missing or Invalid Arguments")
        print("Usage: python collect_images.py [label_name] [number_of_samples]")
        exit(-1)


# This function creates the diractory to save the images
def createDirectory(label_path):
    try:
        os.mkdir(IMAGES_PATH)
    except FileExistsError:
        pass
    try:
        os.mkdir(IMAGES_CLASS_PATH)
    except FileExistsError:
        print(f"{color.CYELLOW}WARNING: {color.CEND} {color.CBLUE}{label_path} {color.CEND} directory already exists. All images collected {color.CBOLD}OVERWRITTEN{color.CEND}")
        global OVERWRITTEN

#Load the saved histogram
def get_hand_hist():
    with open("hand_histogram.pickle","rb") as file:
        hist = pickle.load(file)
    return hist


# This function is copied from genetate_hand_histogram.py file
def back_project(hist,hsv_frame):
    #print(hist)
    dst = cv2.calcBackProject([hsv_frame], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (15,15), 0)
    blur = cv2.medianBlur(blur, 15)
    ret, thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.merge((thresh,thresh,thresh))
    eroded = cv2.erode(thresh,None,iterations=3)
    dilated = cv2.dilate(eroded, None,iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,None)
    return closed



def find_contours(thresh,x,y,w,h):
    thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h,x:x+w]
    contours,hierarchy  = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return thresh,contours


def get_image(contours,thresh):
    save_img = None
    if len(contours) > 0:
        maxCont = max(contours, key=cv2.contourArea)
        # creating the boundry around the object
        if cv2.contourArea(maxCont) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(maxCont)
            save_img = thresh[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2),int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            elif h1 > w1:
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

            save_img = cv2.resize(save_img, (64,64))
    if save_img is None:
        save_image = thresh
    return save_img



def collect_imgs():
    #default webcam
    cap = cv2.VideoCapture(0)
    x, y, w, h = 900, 140, 300, 300
    cap.set(3, 1280) # 3 - PROPERTY index for WIDTH
    cap.set(4, 720) # 4 - PROPERTY index for HEIGHT

    hist = get_hand_hist()

    start = False
    count = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if ret == False:
            print(f"{color.CRED}ERROR:{color.CEND} Frame Not Available Check Camera")
            break
        if count == num_samples:
            cap.release()
            cv2.destroyAllWindows()
            break

        if OVERWRITTEN:
            cv2.putText(frame, "Images are OVERWRITTEN",
                    (950, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        img_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Processing {} / {}".format(count,num_samples),
                 (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press s key on keyboard start saving the images.",
                 (5, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (188,104,20), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press q key on keyboard to quit.",
                 (5, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (188,104,20), 1, cv2.LINE_AA)
        cv2.imshow("Collecting {} images for training".format(label_name), frame)


        if start == True:
            thresh = back_project(hist,img_hsv)

            thresh, contours = find_contours(thresh,x,y,w,h)
            image = get_image(contours,thresh)
            cv2.imshow("thresh",thresh)
            image_path = os.path.join(IMAGES_CLASS_PATH, '{}-{}.jpg'.format(label_name,count + 1))
            cv2.imwrite(image_path, image)
            count += 1

        key = cv2.waitKey(10)

        if key == ord('s'):
            start = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


label_name,num_samples = validate_input()
createDirectory(IMAGES_CLASS_PATH)
collect_imgs()

from keras.models import load_model
import cv2
from collections import OrderedDict
import numpy as np
import sys,os,pickle


IMAGES_PATH = "images"
REV_CLASS_NAME = OrderedDict()

def reverse_classmap():
    global REV_CLASS_NAME
    count = 0
    for directory in os.listdir(IMAGES_PATH):
        REV_CLASS_NAME[count] = str(directory)
        count += 1
def mapper(val):
    return REV_CLASS_NAME[val]

def get_hand_hist():
    with open("hand_histogram.pickle","rb") as file:
        hist = pickle.load(file)
    return hist
def back_project(hist,hsv_frame):
    dst = cv2.calcBackProject([hsv_frame], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (15,15), 0)
    blur = cv2.medianBlur(blur, 15)
    ret, thresh = cv2.threshold(blur,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.merge((thresh,thresh,thresh))
    eroded = cv2.erode(thresh,None,iterations=1)
    dilated = cv2.dilate(eroded, None,iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,None)
    return closed

def find_contours(thresh,x,y,w,h):
    thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h,x:x+w]
    contours,hierarchy  = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return thresh,contours

def get_image(contours,thresh,frame):
    save_img = thresh.copy()

    if len(contours) > 0:
        maxCont = max(contours, key=cv2.contourArea)

        if cv2.contourArea(maxCont) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(maxCont)
            save_img = thresh[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2),int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            elif h1 > w1:
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

    save_img = cv2.resize(save_img, (64,64))

    return save_img

def predict_sign():

    cap = cv2.VideoCapture(0)
    x, y, w, h = 900, 140, 300, 300
    model = load_model("my_cnn_model.h5")
    output_string = ""
    prediction_string = ""
    hist = get_hand_hist()
    while True:
        ret , frame = cap.read()
        frame = cv2.flip(frame, 1)
        img_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        thresh = back_project(hist,img_hsv)
        thresh, contours = find_contours(thresh,x,y,w,h)
        img = get_image(contours,thresh,frame)
        img = img[:, :, np.newaxis]
        pred = model.predict(np.array([img]))
        sign_code = np.argmax(pred[0])
        sign = mapper(sign_code)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Current Prediction => " + sign,
                    (50, 50), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Output: " + output_string,
                    (50, 80), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("TEST", frame)

        k = cv2.waitKey(10)

        if k == ord('s'):
            output_string = output_string + " " + str(sign)
        if k == ord('r'):
            output_string = ""
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


reverse_classmap()
predict_sign()

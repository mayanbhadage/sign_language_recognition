import cv2,sys,os
import numpy as np
from collections import OrderedDict
from keras.models import load_model

IMAGES_PATH = "images"
REV_CLASS_NAME = OrderedDict()
def reverse_classmap():
    global REV_CLASS_NAME
    count = 0
    for directory in os.listdir(IMAGES_PATH):
        REV_CLASS_NAME[count] = str(directory)
        count += 1
reverse_classmap()

filepath = filepath = sys.argv[1]
def mapper(val):
    return REV_CLASS_NAME[val]

model = load_model("my_cnn_model.h5")



img = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
img = img[:, :, np.newaxis]
# predict the move made
pred = model.predict(np.array([img]))
sign_code = np.argmax(pred[0])
sign_name = mapper(sign_code)

print("Predicted: {}".format(sign_name))

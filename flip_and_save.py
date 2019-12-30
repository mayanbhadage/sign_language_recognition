import cv2,sys,os
IMAGES_PATH = "images"
CLASS_NAME = {}
LEN_CLASSES = 0

# Take each directory and convert it into a dict
def getclassmap():
    global CLASS_NAME
    global LEN_CLASSES
    count = 0
    for directory in os.listdir(IMAGES_PATH):
        CLASS_NAME[directory] = count
        count += 1
    LEN_CLASSES = len(CLASS_NAME)
    return count

def generate():
    for dir in CLASS_NAME.keys():
        path = os.path.join(IMAGES_PATH,dir)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if not file.startswith("."):
                    temp = file[::].split(".")[0]
                    file_path = os.path.join(path,file)
                    new_file = "f_"+ temp +".jpg"
                    new_file_path = os.path.join(path,new_file)
                    img = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
                    img = cv2.flip(img,1)
                    cv2.imwrite(new_file_path, img)


def delete_dups():
    for dir in CLASS_NAME.keys():
        path = os.path.join(IMAGES_PATH,dir)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.startswith("f_"):
                    file_path = os.path.join(path,file)
                    os.remove(file_path)

getclassmap()
generate()
#delete_dups()

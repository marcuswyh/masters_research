import cv2
import os
import glob

def getTestFiles():
    pre = os.path.dirname(os.path.realpath(__file__))
    os.chdir(pre+"/test")
    txtname = glob.glob('*.txt')
    imgname = glob.glob('*.jpg')

    textFiles = []
    for f in txtname : 
        if 'flipped' not in f : 
            textFiles.append(os.path.join(pre+"\\test", f))
        else:
            os.remove(os.path.join(pre+"\\test", f))

    imageFiles = []
    for f in imgname : 
        if 'flipped' not in f : 
            imageFiles.append(os.path.join(pre+"\\test", f))
        else:
            os.remove(os.path.join(pre+"\\test", f))

    return textFiles, imageFiles


def getCoordinatesFromText(textfile, image):
    img = cv2.imread(image)
    f = open(textfile, 'r')

    coordinates = f.readline().split()
    stepsLabel = int(f.readline())

    w = int(float(coordinates[3]) * img.shape[1])
    h = int(float(coordinates[4]) * img.shape[0])
    center_x = int(float(coordinates[1]) * img.shape[1])
    center_y = int(float(coordinates[2]) * img.shape[0])

    x1 = int(center_x-(w/2))
    y1 = int(center_y-(h/2))
    x2 = int(center_x+(w/2))
    y2 = int(center_y+(h/2))

    return x1,y1,x2,y2,stepsLabel

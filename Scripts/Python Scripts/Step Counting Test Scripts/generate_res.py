import cv2
import os
import numpy as np
import math
from bbox_convert import getTestFiles, getCoordinatesFromText

def merge_lines(lines, img):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = int((img.shape[1]/100)*2)
    min_angle_to_merge = 5

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        group.append(line)
                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        new_group.append(line2)
            # append new group
            super_lines.append(new_group)

    for group in super_lines:
        super_lines_final.append(merge_lines_segments(group))

    return super_lines_final

def merge_lines_segments(lines):
    if(len(lines) == 1):
        return lines[0]

    line_i = lines[0]
    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        #sort by y
        points = sorted(points, key=lambda point: point[1])
    else:
        #sort by x
        points = sorted(points, key=lambda point: point[0])

    return [points[0], points[len(points)-1]]

def lineMagnitude (x1, y1, x2, y2):
    return math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))

#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine(px, py, x1, y1, x2, y2):
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # closest point does not fall within the line segment, take the shorter distance to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
    return DistancePointLine

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    return min(dist1,dist2,dist3,dist4)

def containsPoint(vertex1, vertex2, point, pos_slope):
    if pos_slope:
        return vertex1[0] <= point[0] <= vertex2[0] and vertex1[1] <= point[1] <= vertex2[1]
    else:
        return vertex1[0] >= point[0] >= vertex2[0] and vertex1[1] <= point[1] <= vertex2[1]

def checkLineWithinBox(x1,x2,y1,y2,point1,point2):
    return (point1[1] < y1 < point2[1] and
            point1[1] < y2 < point2[1] and 
            (point1[0] < x1 < point2[0] or point1[0] < x2 < point2[0]))

def calculateSteps(image, text):
    vertex_x1, vertex_y1, vertex_x2, vertex_y2, STEPS = getCoordinatesFromText(text, image)
    img = cv2.imread(image)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray,cv2.CV_8UC1,0,1,ksize=3)
    ret,edges = cv2.threshold(cv2.bitwise_not(edges), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, (3,3), iterations=1)

    percent = 5
    # minlinelength can be width of detected yolo box in the future
    # minLineLength = vertex_x2 - vertex_x1
    minLineLength = int(((vertex_x2 - vertex_x1)/100) * 60)
    maxLineGap = int((img.shape[1]/100)*percent)
    count=0
    storedLines = []

    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.imshow('',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.imshow('',edges)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.rectangle(img, (vertex_x1, vertex_y1), (vertex_x2, vertex_y2), (0,0,255), 5)
    # cv2.namedWindow("", cv2.WINDOW_NORMAL)
    # cv2.imshow('',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=minLineLength, maxLineGap=maxLineGap)
    for line in lines:   
        for x1,y1,x2,y2 in line:
            angle = np.arctan2((x2-x1),(y2-y1)) * 360 / np.pi
            if angle>=150 and angle<=200:
                storedLines.append([(x1,y1),(x2,y2)])

    storedLines = merge_lines(storedLines, img)

    # detected stairs sample bbox coord
    centroids=[]
    for line in storedLines:
        x1,y1,x2,y2 = line[0][0],line[0][1],line[1][0],line[1][1]
        if (vertex_y1 < y1 < vertex_y2) and (vertex_y1 < y2 < vertex_y2):
            centroid = (x1+int((x2-x1)/2), y1+int((y2-y1)/2))
            centroids.append(centroid)
            # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
            # cv2.circle(img, centroid, 3, (0,0,255), 2)

    if len(centroids) == 0:
        print ("No lines found")
        return None,None
    else:
        points = np.array(centroids)
        # get line of best fit for all centroids
        vx,vy,x,y = cv2.fitLine(points,cv2.DIST_L2,0,0.01,0.01)

        if vx == 0:
            vx = 0.1
        if vy == 0:
            vy = 0.1

        # calculate t values for line fit
        t0 = (0-y)/vy
        t1 = (img.shape[0]-y)/vy

        # if t0 is negative, the line is at a positive slope
        pos_slope = False
        if t0<0:
            pos_slope=True

        # Now find two extreme points on the line to draw line (top to bottom)
        point1 = (x+(t0*vx), y+(t0*vy))
        point2 = (x+(t1*vx), y+(t1*vy))

        # Draw the line of best fit and its bounding box
        cv2.line(img, point1, point2, 255, 2)
        cv2.rectangle(img, point1, point2, 255, 2)

        for line in storedLines:
            x1,y1,x2,y2 = line[0][0],line[0][1],line[1][0],line[1][1]
            centroid = (x1+int((x2-x1)/2), y1+int((y2-y1)/2))
            # stair step is considered if either line points or centroid is within the best line fit's bounding box
            if checkLineWithinBox(x1,x2,y1,y2,point1,point2) or containsPoint(point1,point2,centroid,pos_slope):
                count+=1
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
                cv2.circle(img, centroid, 3, (0,0,255), 2)

        cv2.namedWindow("res", cv2.WINDOW_NORMAL)
        cv2.imshow('res',img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # cv2.imwrite(os.path.join(pre , "res.png"),img)
        accuracy = (count/STEPS)*100
        print ("Steps detected:",count,"    |   Actual steps:",STEPS,"      |   Accuracy:",accuracy)

        # underdetection
        if STEPS > count:
            #return negative value for underdetected steps
            return accuracy, -(STEPS-count)
        # overdetection
        elif count > STEPS:
            return accuracy, count-STEPS
        # perfect detection
        elif count == STEPS:
            return accuracy, None

        return accuracy


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def enhanceContrast(img, p1,p2):
    img =  cv2.imread(img)
    height, width = img.shape[0], img.shape[1]

    cv2.namedWindow("", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    out = np.zeros((height, width, 3), dtype = np.uint8)

    b,c = 80,64
    out = apply_brightness_contrast(img, b, c)

    p1 = np.squeeze(p1)
    p2 = np.squeeze(p2)

    # point1 is at the right
    if p1[0] > p2[0]:
        if p1[0] > width:
            p1[0] = width
        if p2[0] < 0:
            print ("XX")
            p2[0] = 0
        out = out[int(p2[0]):int(p1[0]), int(p1[1]):int(p2[1])]
    # point1 at the left
    else:
        if p1[0] < 0:
            p1[0] = 0
        if p2[0] > width:
            p2[0] = width
        out = out[int(p1[0]):int(p2[0]), int(p1[1]):int(p2[1])]


    print (out.shape)

    cv2.namedWindow("", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("", out)
    cv2.waitKey()
    cv2.destroyAllWindows()

# pre = os.path.dirname(os.path.realpath(__file__))
# fname = 'stairs_4.jpg'
# path = os.path.join(pre, fname)

# calculateSteps(path)

def main():
    texts, images = getTestFiles()
    all_accuracies = []
    normalized_accuracies = []
    overdetection, underdetection, perfectdetection = [],[],[]
    for i in range(len(texts)):
        acc, status = calculateSteps(images[i], texts[i])
        if acc != None:
            all_accuracies.append(acc)

            if status == None:
                perfectdetection.append(status)
                normalized_accuracies.append(acc)
            # overdetection
            elif status > 0:
                overdetection.append(status)
            elif status < 0:
                underdetection.append(status)
                normalized_accuracies.append(acc)

    # acc, status = calculateSteps(images[47], texts[47])


    all_accuracies = np.array(all_accuracies)

    
    print ()
    print ("Detected",len(all_accuracies),"/",len(texts))
    print ("Average accuracy of algorithm:", np.mean(all_accuracies),"%")
    print ("Average Normalized Accuracy of Algorithm (exclude overdetection):",np.mean(np.array(normalized_accuracies)),"%")
    print()
    print ("Number of over-detections:", len(overdetection))
    print ("Number of under-detections:", len(underdetection))
    print ("Number of perfect detections:", len(perfectdetection))
    print()
    print ("Average number of steps over-detected", np.mean(np.array(overdetection)))
    print ("Average number of steps under-detected", np.mean(np.array(underdetection)))

main()



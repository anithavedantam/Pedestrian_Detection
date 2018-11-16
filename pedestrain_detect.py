# Import all the necessary packages
import os
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()

svm_Detector = cv2.HOGDescriptor_getDefaultPeopleDetector()

hog.setSVMDetector(svm_Detector)

# Video Path
vid_path = 'C:/Users/anita/Desktop/My work/pedestrain_detection/pedestrian-dataset/fourway.avi'
#Image Path
img_path = 'C:/Users/anita/Desktop/My work/pedestrain_detection/pedestrian-dataset/Pedestrians.jpg'
final_height = 800.0

# Read the video file
#cap = cv2.VideoCapture(vid_path)
cap = cv2.imread(img_path)
while True:
    #while (cap.isOpened()):
    frame = cap
    #resize the image
    scale = final_height / frame.shape[0]
    frame = cv2.resize(frame, None, fx = scale, fy = scale)


    # Python gradient calculation
    # Read image
    im = frame
    im = np.float32(im) / 255.0

    # Calculate gradient
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Detect multiscale using default detector
    (box, weight) = hog.detectMultiScale(frame, winStride = (8, 8), padding = (32, 32), scale = 1.05)

    for b in box:
        x1, y1, w, h = b
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness = 3, lineType = cv2.LINE_AA)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
    #rects = np.array([[x1, y1, x1 + y2, y1 + y2] for (x1, y1, x2, y2) in box])
    #pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    #for (xA, yA, xB, yB) in pick:
        #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Showing the output in a window
    cv2.imshow("detected frame", frame)
    #cv2.imshow("new", mag)
    #cv2.imshow("new",angle)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closing all the windows
#cap.release()
cv2.destroyAllWindows()

    
        
    

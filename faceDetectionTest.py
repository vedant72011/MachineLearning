import cv2
import matplotlib.pyplot as plt

imagePath = "57323.jpg" #"Screen Shot 2017-07-04 at 11.08.11 AM.png"
cascadePath = "haarcascade_frontalface_default.xml"

#initializing the cascade provided above
faceCascade = cv2.CascadeClassifier(cascadePath)

image = cv2.imread(imagePath) #read the image using cv2
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale because of operations in cv that are done in gray scale

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize = (30,30)
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

#draw a rectangle around the faces
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow("faces found!", image)
#cv2.waitKey(0)
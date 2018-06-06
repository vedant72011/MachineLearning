import cv2
import sys

#firstly specify your cascade
#this one specifically if only for face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

video_capture = cv2.VideoCapture(0)

#this while basically take each frame per second and then analyzes it
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #this to get the gray's out
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detects the face
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 50)
    )

    print "Found {0} faces!".format(len(faces))

    #this will make a rectangal box around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow('Video', frame)
    if 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
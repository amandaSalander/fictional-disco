import cv2

# loading the cascades
face_detection=cv2.CascadeClassifier('module_1_face_recognition/haarcascade_frontalface_default.xml')
eye_detection=cv2.CascadeClassifier('module_1_face_recognition/haarcascade_eye.xml')

print(face_detection)
# define the detection method

def detect(gray, frame):
    faces=face_detection.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        # draw the detected faces on the current frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        # detect eyes on the detected face only of course !!!!
        eyes=eye_detection.detectMultiScale(roi_gray,1.1,3)
        for (x1,y1,w1,h1) in eyes:
            cv2.rectangle(roi_color,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
    
    # return the annotated frame
    return frame



video_capture=cv2.VideoCapture(1)

while True:
    _,frame= video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    result=detect(gray,frame)

    #cv2.imshow("face detection",result)
    cv2.imshow("face detection",frame)

    if (cv2.waitKey(1) & 0xFF== ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
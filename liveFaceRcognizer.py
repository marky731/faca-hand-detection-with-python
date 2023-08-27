import cv2 
from deepface import DeepFace as df
import time
import mediapipe as mp


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #size of the output window
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
counter = 0 
face_match = False
reference_img = cv2.imread('reference.jpg') #put your refenece image to the folder and match up the name

mpHands = mp.solutions.hands # no parantheses!
hands = mpHands.Hands() # only uses RGB 
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0 # for calculation of fps

def check_face(frame):
    try:
        result = df.verify(frame, reference_img)
        return result['verified']
    except ValueError:
        return False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                face_match = check_face(frame.copy()) #check the face (used copy module to avoide possible issues)
            except ValueError:  # if no face detected, continue 
                pass
        counter += 1
        
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert it to gray
            haar_cascade = cv2.CascadeClassifier('haar_face.xml') # required data file 
            faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=7) #detect position of the face in the frame
         
            for (x,y,w,h) in faces_rect:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)  #draw green rectangle
        else:
            cv2.putText(frame, "NOT MATCH!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3) 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_cascade = cv2.CascadeClassifier('haar_face.xml')
            faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=9)

            for (x,y,w,h) in faces_rect:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), thickness=2)  # draw red rectangle if face is not mathced to reference

        if not ret:  # if input from camare is not recieved
            print("not success!")
            break

        image_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert it to RGB form
        result = hands.process(image_RGB)  # this modul reqiures RGB 
        print(result.multi_hand_landmarks) # print the coordinates of hand(s)

        if result.multi_hand_landmarks: # if any hand is detected 
            for hand_landmarks in result.multi_hand_landmarks:
                for id,lm in enumerate(hand_landmarks.landmark): # id is points of fingers palm etc 
                    # print(id,lm)
                    h, w, c = frame.shape 
                    cx, cy = int(lm.x * w), int(lm.y*h)
                    if id == 4: # there are 21 points, pick which ever you want
                        cv2.circle(frame,(cx,cy), 15, (255,0,255), cv2.FILLED) # draw a purple circl 
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS) # not rgb because we display otg img


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)


        cv2.imshow('video', frame)


    key = cv2.waitKey(1)
    if key == ord('q'): # press q when you want to terminate
        break

cv2.destroyAllWindows()

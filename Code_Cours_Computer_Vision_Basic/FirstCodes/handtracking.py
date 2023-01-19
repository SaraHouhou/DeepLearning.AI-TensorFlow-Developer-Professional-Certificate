import cv2
import mediapipe as mp

# Run a webcam
cap= cv2.VideoCapture(0)

# create an object from a hand from a hand class

mpHands= mp.solutions.hands
# inside the object han we have different parameter
#static_image mode will be always false in order to do the static and the dynamic # static_image_mode=False , maximum_num_hands=2
hands=mpHands.Hands()

# instancier l'objet de design in order to draw the rgb image with points
mpDraw = mp.solutions.drawing_utils

while True: 

    success, img = cap.read()

    #send our rgb imaage
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    # # open the detected image 
    # # print(results.multi_hand_land_marks) 

    if results.multi_hand_landmarks:
    #     # handlms may be the hand number 0 or 1
        for handlms in results.multi_hand_landmarks:
            # Case one draw only points
          #mpDraw.draw_landmarks(img,handlms)
           # Case 2 ; Draw arcs between points
           # each id has a land marrk on the hand and each land mark has x, y, z decimil place if its is a 0 is a bottom and if it is 1 is a landup ..etc
           for id, lm in enumerate(handlms.landmark):
            #check ou the hight, the wisth and the channels of our image
            hight, width, channels = img.shape
            # find the positions
            cx, cy=int(lm.x*width), int(lm.y*hight)
            #print (id, cx, cy), pour chaque id nous avons cx et cy
            if id==0:
                cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED) # WE GIVE THE IMAGE? THE POSITION X AND Y OF THE ID, la taille DE ROND? LA COLOR DE ROND ET L4ORDER DE REMPLISSAGE
            mpDraw.draw_landmarks(img,handlms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    # # close the window
    if cv2.waitKey(1) == ord('q'): # wait 1 ms after pressing q in the keyboard to close it
        break

    #cv2.destroyAllWindows()
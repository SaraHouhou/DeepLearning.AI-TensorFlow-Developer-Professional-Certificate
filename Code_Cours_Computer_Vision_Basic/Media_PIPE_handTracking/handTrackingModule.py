
import cv2
import mediapipe as mp

class handDetector():
    #def __init__(self, mode=False, maxHands = 2, model_complexity = 1, detectionCon=0.5, trackCon=0.5):

    def __init__(self, mode=False, maxHands = 2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
       # self.model_complexity= model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
# create an object from a hand from a hand class
        self.mpHands= mp.solutions.hands
# inside the object han we have different parameter
#static_image mode will be always false in order to do the static and the dynamic # static_image_mode=False , maximum_num_hands=2
        self.hands=self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)

        #self.hands=self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
# instancier l'objet de design in order to draw the rgb image with points
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img, draw=True):
        #send our rgb imaage
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
            # # open the detected image 
            # # print(results.multi_hand_land_marks) 
        if self.results.multi_hand_landmarks:
            #     # handlms may be the hand number 0 or 1
            for handlms in self.results.multi_hand_landmarks:
                    # Case one draw only points
                #mpDraw.draw_landmarks(img,handlms)
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPositionforOneParticularHand(self, img, handNo=0, draw=True): #HandnUMBER 0
        # restor the lm positions
        lmList= []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
             # Case 2 ; Draw arcs between points
    #         # each id has a land marrk on the hand and each land mark has x, y, z decimil place if its is a 0 is a bottom and if it is 1 is a landup ..etc
            for id, lm in enumerate(self.handlms.landmark):
             #check ou the hight, the wisth and the channels of our image
                 hight, width, channels = img.shape
    #             # find the positions
                 cx, cy=int(lm.x*width), int(lm.y*hight)
    #             #print (id, cx, cy), pour chaque id nous avons cx et cy
                 lmList.append([id, cx, cy])
                 if draw:
                     cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED) # WE GIVE THE IMAGE? THE POSITION X AND Y OF THE ID, la taille DE ROND? LA COLOR DE ROND ET L4ORDER DE REMPLISSAGE
        return lmList

def main():
    cap=cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success, img=cap.read()
        img= detector.findHands(img)
        lmList = detector.findPositionforOneParticularHand(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.imshow("Image", img)
        # # close the window
        if cv2.waitKey(1) == ord('q'): # wait 1 ms after pressing q in the keyboard to close it
            break


    # 
if __name__== "__main__":
    main()


#     There are some arguments that we need to add in new version of python and mediapipe, i think :
# 1. In hand tracking (static image, max number of hands, model complexity, min detection confidence, min tracking confidence)
# 2. In pose estimation (static image , model_complexity, upper_body_only, smooth_landmarks, detection confidence, tracking confidence)
# 3. In face mesh (static image, max. no. of faces, redefine landmarks, min. detection confidence, min. tracking confidence)
# 4.  To draw the landmarks in face mesh : mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

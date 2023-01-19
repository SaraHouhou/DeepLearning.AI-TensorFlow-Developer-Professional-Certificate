import cv2
import mediapipe as mp

class poseDetector():

    def __init__(self, Static_image_mode= False, upper_body_only= False , smooth_landmarks=True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):

        self.Static_image_mode= Static_image_mode
        self.upper_body_only= upper_body_only 
        self.smooth_landmarks=smooth_landmarks
        self.min_detection_confidence = min_detection_confidence 
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose= self.mpPose.Pose(self.Static_image_mode, self.upper_body_only, self.smooth_landmarks, self.min_detection_confidence, self.min_tracking_confidence)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_land_marks) 
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img



def main():

    cap = cv2.videoCapture('PoseVideo/1.mp4')
# instancier l'objet de design in order to draw the rgb image with points
    detector=poseDetector()
    while True: 
        sucess, img = cap.read()
        img=detector.findPose(img)

    
        #     for id, lm in enumerate (results.pose_landmarks.landmark):
        #         #check ou the hight, the wisth and the channels of our image
        #             hight, width, channels = img.shape
        # #             # find the positions
        #             cx, cy=int(lm.x*width), int(lm.y*hight)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
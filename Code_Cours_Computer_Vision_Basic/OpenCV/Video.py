import numpy as np
import cv2


# give 0 one of camera plus selon ce qu'on a 

cap =cv2.VideoCapture(0)

# while True: 
#     ret, frame = cap.read()
#     # ret: elle tu done false s'il ya un probleme dans le camera

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) == ord('q'): # wait 1 ms after pressing q in the keyboard to close it
#         break

# cap.release()
# cv2.destroyAllWindows()

#Duppliquer la cam en 4 photo sur le meme frame

# while True: 
#     ret, frame = cap.read()
#     width= int(cap.get(3))
#     height = int(cap.get(4))

#     image= np.zeros(frame.shape, np.uint8)
#     smaller_frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
#     image[:height//2, :width//2]= smaller_frame
#     image[height//2:, :width//2]= smaller_frame
#     image[:height//2, width//2:]= smaller_frame
#     image[height//2:, width//2:]= smaller_frame

#     # ret: elle tu done false s'il ya un probleme dans le camera

#     cv2.imshow('frame', image)

#     if cv2.waitKey(1) == ord('q'): # wait 1 ms after pressing q in the keyboard to close it
#         break

# cap.release()
# cv2.destroyAllWindows()





#Duppliquer la cam en 4 photo sur le meme frame, et rotation chaque frame dans une coté

while True: 
    ret, frame = cap.read()
    width= int(cap.get(3))
    height = int(cap.get(4))

    image= np.zeros(frame.shape, np.uint8)
    smaller_frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    image[:height//2, :width//2]= cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, :width//2]= smaller_frame
    image[:height//2, width//2:]= cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, width//2:]= smaller_frame

    # ret: elle tu done false s'il ya un probleme dans le camera

    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'): # wait 1 ms after pressing q in the keyboard to close it
        break

cap.release()
cv2.destroyAllWindows()
# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mount_everest.jpg')

# resizing the mountain image as 640 X 480
mountain.resize(3, 640)
mountain.resize(4, 480)

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([0, 120, 50])
        upper_bound = np.array([10, 255, 255])

        # thresholding image
        mask_1 = cv2.inRange(frame_rgb, lower_bound, upper_bound)
        # inverting the mask

        mask_2 = np.flip(mountain, axis =1)

        
        # bitwise and operation to extract foreground / person
        mask_2 = cv2.bitwise_not(mask_1)

        result1 = cv2.bitwise_and(mask_1, mask_1, mask = mask_2)

        result2 = cv2.bitwise_and(mask_2, mask_2, mask = mask_1)
        # final image
        final_img = cv2.addWeighted(result1, 1, result2, 1, 0 )
        camera.read(final_img)
        # show it
        cv2.imshow('frame' , final_img)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()

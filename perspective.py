import numpy as np
import cv2 as cv

import time

"""
img1 = cv.imread("2.jpg")
img2 = cv.imread("1.jpg")
# Define matching points
corners1 = [[[221.0, 362.0]],
           [[191.0, 41.0]],
           [[607.0, 118.0]],
           [[468.0, 416.0]]]
corners1 = np.array(corners1)
corners2 = [[[224.0, 60.0]],
           [[534.0, 47.9]],
           [[529.0, 302.0]],
           [[231.0, 284.0]]]
corners2 = np.array(corners2)
"""

img1 = cv.imread("4.jpg")
img2 = cv.imread("3.jpg")

# Define matching points
corners1 = [[[209.0, 242.0]],
           [[450.0, 240.0]],
           [[554.0, 357.0]],
           [[99.0, 356.0]]]
corners1 = np.array(corners1)

corners2 = [[[102.0, 58.0]],
           [[555.0, 58.0]],
           [[553.0, 354.0]],
           [[102.0, 352.0]]]
corners2 = np.array(corners2)

# Calculate homography
t0 = time.time()
H, _ = cv.findHomography(corners1, corners2)
img1_warp = cv.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]), flags=cv.INTER_LINEAR)

t1 = time.time()

total = t1-t0
print(total)


cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.imshow("warp", img1_warp )


cv.waitKey(0)



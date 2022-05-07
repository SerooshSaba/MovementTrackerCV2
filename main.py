import math
import time
import cv2
import numpy
import numpy as np

from Vec2f import Vec2f
from Vehicle import Vehicle

font = cv2.FONT_HERSHEY_SIMPLEX
frame = 1

# Original
corners1 = np.array([[[210.0, 244.0]],
                    [[450.0, 241.0]],
                    [[554.0, 357.0]],
                    [[100.0, 356.0]]])
# Distorted to fit
corners2 = np.array([[[193.0, 165.0]],
                    [[451.0, 164.0]],
                    [[473.0, 336.0]],
                    [[173.0, 337.0]]])

# Calculate homography matrix for distortion of image
H, _ = cv2.findHomography(corners1, corners2)

# Setup orb feature detector and brute force matcher
orb = cv2.ORB_create( edgeThreshold=100, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=50)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Store previous frame keypoints and descriptors
_kp = None
_des = None

CAMERA_X = 640 / 2
CAMERA_Y = 480 + 50

ROTATION = 0



# Empty image for tracking
blank_image = np.zeros((480,640,3), np.uint8)
dir = Vec2f(0,-1)
vehicle = Vehicle()

cap = cv2.VideoCapture('video.mp4')
while (cap.isOpened()):

    ret, image = cap.read()

    if ret == True:
        compute = ( frame % 2 == 0 )
        image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        image = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

        if compute:
            kp, des = orb.detectAndCompute(image, None)

            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray = np.float32(gray)
            """
            img - Input image. It should be grayscale and float32 type.
            blockSize - It is the size of neighbourhood considered for corner detection
            ksize - Aperture parameter of the Sobel derivative used.
            k - Harris detector free parameter in the equation.
            """
            #dst = cv2.cornerHarris(gray, 5, 1, 0.04)
            #image[dst > 0.01 * dst.max()] = [0, 0, 255]

            # Match features
            if _kp is not None and _des is not None:

                #image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
                #image = cv2.drawKeypoints(image, _kp, None, color=(0, 0, 255), flags=0)

                # des -> queryIdx | _des -> trainIdx
                matches = bf.match(des, _des)

                length_filtered = []
                # Filter matches for distance
                for match in matches:

                    p1 = kp[match.queryIdx].pt # Descriptor from current image
                    p2 = _kp[match.trainIdx].pt # Previous

                    x, y = p1
                    _x, _y = p2
                    xd, yd = x - _x, y - _y

                    length = math.sqrt(math.pow(xd,2)+math.pow(yd,2))
    
                    # Remove matches that are too far apart
                    if length != 0 and length < 50:
                        length_filtered.append(match)
                        #start = ( int(_x), int(_y) )
                        #end = ( int(x), int(y) )
                        #image = cv2.line(image, start, end, (0,0,255), 2)

                # Find angle of all length filtered matches
                angles = []
                for match in length_filtered:
                    p1 = kp[match.queryIdx].pt
                    p2 = _kp[match.trainIdx].pt
                    x, y = p1
                    _x, _y = p2
                    xd = x - _x
                    yd = y - _y

                    vec = np.array([xd,yd])
                    angles.append( np.arctan2(*vec.T[::-1]) * 180 / np.pi )

                final_filter = []

                # Remove angles that deviate
                meana = np.mean(angles)
                stda = np.std(angles)
                min = meana - 0.5 * stda
                max = meana + 0.5 * stda

                i = 0
                while i < len(length_filtered):
                    a = angles[i]
                    if max > a > min:
                        final_filter.append( length_filtered[i] )
                    i += 1

                if len(final_filter):

                    cur_coord = []
                    prv_coord = []

                    # Render final filtered matches
                    for match in final_filter:
                        p1 = kp[match.queryIdx].pt
                        p2 = _kp[match.trainIdx].pt
                        x, y = p1
                        _x, _y = p2

                        cur_coord.append([x, y])
                        prv_coord.append([_x, _y])

                        start = (int(_x), int(_y))
                        end = (int(x), int(y))
                        image = cv2.line(image, start, end, (255, 0, 0), 2)

                       
                        start = (int(_x), int(y))
                        end = (int(x), int(y))
                        image = cv2.line(image, start, end, (0, 0, 255), 2)
                        

                    # Calculate rotation in frame
                    rotation_in_frame = 0

                    # Calculate rotation in frame
                    i = 0
                    while i < len(cur_coord):

                        x, y = cur_coord[i]
                        _x, _y = prv_coord[i]

                        # Find amount to translate prv kp to center
                        prv_to_center = CAMERA_X - _x

                        start = (int(_x + prv_to_center), int(y))
                        end = (int(x + prv_to_center), int(y))
                        image = cv2.line(image, start, end, (0, 0, 255), 2)

                        p2 = np.array([ _x + prv_to_center, y ])
                        p3 = np.array([ x + prv_to_center, y ])

                        p1 = np.array([CAMERA_X,CAMERA_Y])
                        v1 = p1 - p2
                        v2 = p1 - p3

                        angle = np.arccos(  np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )

                        if ( x > _x ):
                            angle *= -1

                        rotation_in_frame += angle

                        i += 1

                    rotation_in_frame /= len(final_filter)
                    dir.rotate(rotation_in_frame)
                    ROTATION += np.rad2deg(rotation_in_frame)

                    # Render coordinate systems
                    """
                    i = 0
                    while i < len(coord1):
                        j = 0
                        while j < len(coord1):
                            if i != j:
                                p1x = int(coord1[i][0])
                                p1y = int(coord1[i][1])
                                p2x = int(coord1[j][0])
                                p2y = int(coord1[j][1])
                                image = cv2.line( image, (p1x, p1y), (p2x, p2y), (0,255,0), 1 )
                            j += 1
                        i += 1
                    i = 0
                    while i < len(coord2):
                        j = 0
                        while j < len(coord2):
                            if i != j:
                                p1x = int(coord2[i][0])
                                p1y = int(coord2[i][1])
                                p2x = int(coord2[j][0])
                                p2y = int(coord2[j][1])
                                image = cv2.line( image, (p1x, p1y), (p2x, p2y), (0,0,255), 1 )
                            j += 1
                        i += 1
                    """

                    # FIND ROTATION AND TRANSLATION BETWEEN COORDINATE SYSTEMS - - - - - - - - - - -
                    A = np.array(prv_coord)
                    B = np.array(cur_coord)
    
                    # Find centroid A
                    L = A.shape[0]
                    Asx, Asy = np.sum(A[:,0]) / L, np.sum(A[:,1]) / L
                    Acentroid = np.array([Asx,Asy])
    
                    # Find centroid B
                    L = B.shape[0]
                    Bsx, Bsy = np.sum(B[:,0]) / L, np.sum(B[:,1]) / L #TODO < sum/l can be changed with mean
                    Bcentroid = np.array([Bsx,Bsy])
                
                    # Draw centroid
                    acx = int(Acentroid[0])
                    acy = int(Acentroid[1])
                    image = cv2.rectangle(image, (acx, acy), (acx + 5, acy + 5), (0, 255, 0), -1)
                    bcx = int(Bcentroid[0])
                    bcy = int(Bcentroid[1])
                    image = cv2.rectangle(image, (bcx, bcy), (bcx + 5, bcy + 5), (0, 0, 255), -1)
                    
                    # Calculate translation of feature coordinates

                    T = B - A
                    tx, ty = np.mean(T[:,0]), np.mean(T[:,1]) # TODO < can use centroids to calc translation

                    #vehicle.transX(tx)
                    vehicle.transY(ty)
                    vehicle.rotate(rotation_in_frame)
                    blank_image = vehicle.render(blank_image, 640/2, 480/2 )
         
                    # Translate systems to origin
                    A = A - Acentroid
                    B = B - Bcentroid
                    
                    # DRAW Translate to center
                    tx = image.shape[1] / 2
                    ty = image.shape[0] / 2
                    transmat = np.array([tx,ty])
                    A = A + transmat
                    B = B + transmat
                    for point in A:
                        px = int(point[0])
                        py = int(point[1])
                        cv2.circle(image,(px,py), 3, (255,255,255), -1 )
                    for point in B:
                        px = int(point[0])
                        py = int(point[1])
                        cv2.circle(image,(px,py), 3, (0,0,255), -1 )


            _kp = kp
            _des = des
            cv2.imshow('Frame', image)
            cv2.imshow('track', blank_image)

        frame += 1
        #time.sleep(0.1)

    else:
        break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
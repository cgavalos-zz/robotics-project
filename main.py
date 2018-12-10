from DobotSerialInterface import DobotSerialInterface
import time

import cv2
import threading
import numpy as np

class Dobot:
    def __init__(self, port):
        self.interf = DobotSerialInterface('COM' + str(port))
        self.interf.set_speed()
        self.interf.set_playback_config()

    def debug(self):
        print('Joint Angles: ' + str(self.interf.current_status.angles))

    def jmove(self, base, rear, front, rot):
        self.interf.send_absolute_angles(
            base, rear, front, rot,
            move_mode=DobotSerialInterface.MOVE_MODE_JOINTS)
        time.sleep(0.25)

    def zero(self):
        self.jmove(0, 0, 0, 0)

    def dzero(self):
        self.zero()
        self.debug()

    def djmove(self, base, rear, front, rot):
        self.jmove(base, rear, front, rot)
        self.debug()

def r2():
    d = Dobot('6')
    d.dzero()
    d.djmove(0, 45, 0, 0)
    d.dzero()
    d.djmove(0, 0, 45, 0)
    d.dzero()
    d.djmove(0, 0, 0, 45)
    d.dzero()

def capshow(n):
    cap = cv2.VideoCapture(n)
    while True:
        ret, img = cap.read()
        rows, cols, depth = img.shape

        #M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        #dst = cv2.warpAffine(img,M,(cols,rows))

        ratio = 2
        lowv = 100

        circs = True

        if circs:
            for c in object_centroids(img, lowv, ratio):
                cv2.circle(img, c, 10, (255, 0, 0), -1)

            cv2.imshow('circs', img)
        else:
            cv2.imshow('blobs', picture_to_blobs(img, lowv, ratio))

        key = cv2.waitKey(100)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def acapshow(n):
    thr = threading.Thread(target=capshow, args=[n], kwargs={})
    thr.start()

def fill_closed(closed_img):
    th, im_th = cv2.threshold(closed_img, 220, 255, cv2.THRESH_BINARY);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    return im_th | im_floodfill_inv

def picture_to_blobs(picture, low, ratio):
    # Get edges using hue? (could be saturation but probs not)
    high = ratio * low
    use_hsv = True
    if use_hsv:
        hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(hsv[:, :, 1], low, high)
    else:
        edges = cv2.Canny(picture, low, high)

    # Find contours of edges.
    im2, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw em thicc to close holes in curves.
    cv2.drawContours(im2, contours, -1, (255, 0, 0), 10)

    # Fill closed curves for blobs.
    return fill_closed(im2)

def object_centroids(img, lowv, ratio):
    blobs = picture_to_blobs(img, lowv, ratio)
    second_edge = cv2.Canny(blobs, lowv, lowv * ratio)
    im, contours, hierarchy = cv2.findContours(
        second_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    M = [cv2.moments(contour) for contour in contours]
    fM = filter(lambda m: m['m00'] != 0, M)

    Cx = [int(m['m10'] / m['m00']) for m in fM]
    Cy = [int(m['m01'] / m['m00']) for m in fM]
    return zip(Cx, Cy)

def edge_test(lowv):
    cap = cv2.VideoCapture(1)
    ret, img = cap.read()

    ratio = 2

    for c in object_centroids(img, lowv, ratio):
        cv2.circle(img, c, 10, (255, 0, 0), -1)

    cv2.imshow('circles', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def r2():
    acapshow(1)
    return Dobot(6)

if __name__ == '__main__':
    capshow(1)

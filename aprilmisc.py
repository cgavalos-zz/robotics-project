import apriltag
import cv2

def get_results(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = apriltag.Detector()
    return (img, detector.detect(gray))

def annotate_image(img, rs):
    first = True
    for r in rs:
        if first:
            first = False
        c = r.center
        cv2.circle(img, tuple([int(x) for x in c]), 2, (r.tag_id, r.tag_id, r.tag_id))

        corns = r.corners

        i = 0
        for corner in corns:
            if i == 0:
                c = (255, 0, 0)
            elif i == 1:
                c = (0, 255, 0)
            elif i == 2:
                c = (0, 0, 255)
            else:
                c = (0, 255, 255)

            cv2.circle(img, tuple([int(x) for x in corner]), 2, c)
            i += 1

    return img

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

def find_result_by_tag_id(results, id):
    return next((t for t in results if t.tag_id == id), None)

def debugannotate(img, res):
    img = annotate_image(img, res)

    intuple = lambda a: tuple([int(i) for i in a])

    for r in res:
        up = r.corners[0] - r.corners[3]

        cv2.line(img, intuple(r.center), intuple(r.center + up), (255, 255, 255))
        cv2.putText(img, str(r.tag_id), intuple(r.center), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2, cv2.LINE_AA)

    return img

def aprildebug(n):
    cap = cv2.VideoCapture(n)

    while True:
        ret, img = cap.read()

        img, res = get_results(img)

        img = annotate_image(img, res)

        intuple = lambda a: tuple([int(i) for i in a])

        for r in res:
            up = r.corners[0] - r.corners[3]

            cv2.line(img, intuple(r.center), intuple(r.center + up), (255, 255, 255))
            cv2.putText(img, str(r.tag_id), intuple(r.center), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2, cv2.LINE_AA)


        cv2.imshow('apriltag-debug', img)

        key = cv2.waitKey(1000 / 30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

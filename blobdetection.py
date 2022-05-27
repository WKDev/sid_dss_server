
import cv2
import numpy as np


def blobdetector(en, source):
    # Load image
    # image = cv2.imread('blob.png', 0)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.9

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()

        cvted_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if ret:

            if en:
                # Detect blobs
                keypoints = detector.detect(cvted_frame)

                # Draw blobs on our image as red circles
                blank = np.zeros((1, 1))
                ret_detected = cv2.drawKeypoints(cvted_frame, keypoints, blank, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                number_of_blobs = len(keypoints)
                text = "Number of Circular Blobs: " + str(len(keypoints))
                # cv2.putText(blobs, text, (20, 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

                # cv2.imshow("Filtering Circular Blobs Only", blobs)

                ret, img = cv2.imencode('.jpg', ret_detected)

                byte_img = img.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type:image/jpeg\r\n'
                    b'Content-Length: ' + f"{len(byte_img)}".encode() + b'\r\n'
                    b'\r\n' + bytearray(byte_img) + b'\r\n')

            else:
                ret, img = cv2.imencode('.jpg', cvted_frame)

                byte_img = img.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type:image/jpeg\r\n'
                    b'Content-Length: ' + f"{len(byte_img)}".encode() + b'\r\n'
                    b'\r\n' + bytearray(byte_img) + b'\r\n')


                




if __name__ == "__main__":
    blobdetector("http://192.168.0.156:5000/live/2")

import cv2
import numpy as np
import math

def main(args=None):
    # camera calibration 결과값 내부 파라미터 load
    with np.load("camera_calibration_data.npz") as data:
        mtx = data["mtx"]
        dist = data["dist"]

    marker_length = 0.096    # aruco makre의 실제 길이 (m)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)   # corners = 2d 픽셀 좌표

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], marker_length)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                rotation_degree = rotMat2degree(rotation_matrix)

                print(f'Rotation : {rotation_matrix}, Translation : {tvecs[i]}')
                print(f'angle : {rotation_degree}')

        cv2.imshow("Aruco Marker Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

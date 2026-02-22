import cv2
import pyrealsense2 as rs
import numpy as np
import math

# ==========================================
# [사용자 설정 영역] 이 부분을 본인 환경에 맞게 수정하세요!
# ==========================================

# 1. 아루코 마커의 한 변의 길이 (단위: 미터)
MARKER_SIZE = 0.096  # 예: 10cm라면 0.1

# 2. 로봇으로 측정한 마커 정중앙의 좌표 (단위: 미터)
# 로봇 TCP를 마커 중앙에 찍었을 때의 좌표 (X, Y, Z)
MARKER_IN_ROBOT_FRAME = np.array([0.271, 0.0, 0.130]) # 예시값: X=450mm 지점

# 3. 마커의 방향 보정 (로봇 좌표계 기준)
# 마커를 로봇 X, Y축과 평행하게 붙였다면 회전은 Identity 행렬에 가깝습니다.
# 만약 마커가 돌아가 있다면 이 부분을 수정해야 하지만, 일단 평행하다고 가정합니다.
# (필요시 Z축 회전 변환 추가 가능)
ROBOT_TO_MARKER_ROTATION = np.eye(3) 

# ==========================================

def get_transform_matrix(rvec, tvec):
    """ rvec, tvec를 4x4 변환 행렬로 변환하는 함수 """
    mat = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    mat[:3, :3] = R
    mat[:3, 3] = tvec.flatten()
    return mat

def main():
    # 1. RealSense 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 스트리밍 시작
    profile = pipeline.start(config)

    # 내장 파라미터(Intrinsics) 가져오기
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(intrinsics.coeffs)

    # 2. ArUco 설정 (DICT_5X5_250 등 본인이 쓰는 마커 종류로 변경 필요)
    # 사진 속 마커는 5x5 또는 4x4로 보입니다. (일단 5x5로 시도)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # 계산된 최종 변환 행렬을 저장할 변수
    T_base_to_camera = None

    print("--- ArUco 마커를 찾아 캘리브레이션을 시작합니다 ---")
    print("마커가 보이면 자동으로 변환 행렬을 계산합니다.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 프레임 정렬 (Depth -> Color)
            align = rs.align(rs.stream.color)
            frames = align.process(frames)
            aligned_depth_frame = frames.get_depth_frame()
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # ArUco 마커 검출
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None and T_base_to_camera is None:
                # 마커가 발견되었고, 아직 행렬 계산 전이라면
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_matrix, dist_coeffs)
                
                # 첫 번째 발견된 마커(인덱스 0)를 기준으로 삼음
                rvec = rvecs[0]
                tvec = tvecs[0]

                # 시각화 (축 그리기)
                cv2.drawFrameAxes(color_image, cam_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                # ---------------------------------------------------------
                # [핵심] 좌표 변환 행렬 계산 (동차 변환)
                # ---------------------------------------------------------
                
                # 1. T_camera_to_marker (카메라 기준 마커의 위치)
                T_cam_marker = get_transform_matrix(rvec, tvec)
                
                # 2. T_base_to_marker (로봇 베이스 기준 마커의 위치 - 사용자가 입력한 값)
                T_base_marker = np.eye(4)
                T_base_marker[:3, :3] = ROBOT_TO_MARKER_ROTATION
                T_base_marker[:3, 3] = MARKER_IN_ROBOT_FRAME
                
                # 3. T_base_to_camera (우리가 구하고 싶은 것: 로봇 베이스 기준 카메라의 위치)
                # 수식: T_base_cam = T_base_marker * inv(T_cam_marker)
                T_base_to_camera = np.dot(T_base_marker, np.linalg.inv(T_cam_marker))

                print("\n✅ 캘리브레이션 완료!")
                print("--- 계산된 T_base_to_camera 행렬 ---")
                print(np.array_str(T_base_to_camera, precision=3, suppress_small=True))
                print("------------------------------------\n")

            # 마커가 감지되면 테두리 그리기
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            # 마우스 클릭 이벤트 처리를 위한 함수
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if T_base_to_camera is None:
                        print("⚠️ 아직 마커를 찾지 못해 캘리브레이션이 안 되었습니다.")
                        return
                    
                    # 1. 픽셀(u,v) -> 카메라 좌표계(Xc, Yc, Zc) 변환
                    depth_val = aligned_depth_frame.get_distance(x, y) # 미터 단위
                    if depth_val <= 0:
                        print("⚠️ 깊이 값을 읽을 수 없습니다.")
                        return

                    point_camera = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_val)
                    
                    # 동차 좌표로 변환 [x, y, z, 1]
                    point_camera_homo = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])

                    # 2. 카메라 좌표계 -> 로봇 좌표계 변환 (행렬 곱셈)
                    point_robot_homo = np.dot(T_base_to_camera, point_camera_homo)

                    final_x = point_robot_homo[0] * 1000 # mm 변환
                    final_y = point_robot_homo[1] * 1000
                    final_z = point_robot_homo[2] * 1000

                    print(f"클릭 좌표(픽셀): ({x}, {y}) / Depth: {depth_val:.3f}m")
                    print(f"🎯 로봇 목표 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}")
                    print("--------------------------------------------------")

            cv2.namedWindow('RealSense ArUco', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('RealSense ArUco', mouse_callback)
            cv2.imshow('RealSense ArUco', color_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
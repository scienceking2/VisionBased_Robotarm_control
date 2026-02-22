import cv2
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import time

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import DR_init
# DSR 라이브러리가 로컬에 있다고 가정
from realsense.gripper_drl_controller import GripperController 

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 50, 50

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")

        self.bridge = CvBridge()
        self.get_logger().info("ROS 2 구독자 설정을 시작합니다...")

        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        
        # 캘리브레이션으로 구한 4x4 변환 행렬 (Camera to Robot Base)
        self.T_base_cam = np.array(
            [
                [-0.017, 0.885, -0.465, 0.848],
                [0.999, 0.002, -0.032, 0.064],
                [-0.027, -0.466, -0.885, 1.06],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        
        # 📌 수정: 토픽 리스트 기반, 네임스페이스('/camera/camera/')가 붙은 토픽으로 통일
        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'  # 토픽 리스트에 존재하는 네임스페이스
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw' # 토픽 리스트에 존재하는 네임스페이스
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info' # 토픽 리스트에 존재하는 네임스페이스
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("컬러/뎁스/카메라정보 토픽 구독 대기 중...")
        self.get_logger().info("화면이 나오지 않으면 RealSense Launch 명령어를 확인하세요.")

        # --- 로봇 및 그리퍼 초기화 (생략 없이 동일) ---
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait 
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("Waiting for service /dsr01/drl/drl_start...")
            wait(2) 
            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed. Exiting.")
                raise Exception("Gripper initialization failed")
            self.get_logger().info("그리퍼를 활성화합니다...")
            self.gripper_is_open = True
            self.gripper.move(700)
            
        except Exception as e:
            self.get_logger().error(f"An error occurred during gripper setup: {e}")
            pass 

        self.get_logger().info("RealSense ROS 2 구독자와 로봇 컨트롤러가 초기화되었습니다.")
        # ---------------------------

    def synced_callback(self, color_msg, depth_msg, info_msg):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge 변환 오류: {e}")
            return

        # Intrinsics 설정은 변화 없음
        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            
            if info_msg.distortion_model == 'plumb_bob' or info_msg.distortion_model == 'rational_polynomial':
                self.intrinsics.model = rs.distortion.brown_conrady
            else:
                self.intrinsics.model = rs.distortion.none
            
            self.intrinsics.coeffs = list(info_msg.d)
            self.get_logger().info("카메라 내장 파라미터(Intrinsics) 수신 완료.")

    # stop_camera, terminate_gripper는 동일

    def stop_camera(self):
        pass

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    def mouse_callback(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_cv_depth_mm is None or self.intrinsics is None:
                self.get_logger().warn("아직 뎁스 프레임 또는 카메라 정보가 수신되지 않았습니다.")
                return

            try:
                depth_mm = self.latest_cv_depth_mm[v, u]
            except IndexError:
                self.get_logger().warn(f"클릭 좌표(u={u}, v={v})가 이미지 범위를 벗어났습니다.")
                return
            
            if depth_mm == 0:
                print(f"({u}, {v}) 지점의 깊이를 측정할 수 없습니다 (값: 0).")
                return

            depth_m = float(depth_mm) / 1000.0

            point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_m)

            # 📌 로봇 좌표계 변환 로직 보강 (캘리브레이션 행렬 적용)
            # RealSense 3D 좌표계에서 로봇 베이스 좌표계로 변환
            # 행렬 T_base_cam은 [미터(m)] 단위를 기준으로 하므로,
            # 계산 후 x, y, z 값을 mm 단위로 변환해줍니다.
            
            cam_point = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
            robot_point = np.dot(self.T_base_cam, cam_point)
            
            # 로봇 좌표계 (mm 단위로 변환)
            final_x = robot_point[0] * 1000.0
            final_y = robot_point[1] * 1000.0
            final_z = robot_point[2] * 1000.0 - 10.0
            # --------------------------------------------------------

            if(final_z <= 150):
                final_z = 150

            print("--- 변환된 최종 3D 좌표 ---")
            print(f"픽셀 좌표: (u={u}, v={v}), Depth: {depth_m*1000:.1f} mm")
            print(f"로봇 목표 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}\n")

            self.move_robot_and_control_gripper(final_x, final_y, final_z)
            print("=" * 50)

    # move_robot_and_control_gripper 함수 및 main 함수는 동일

    def move_robot_and_control_gripper(self, x, y, z):
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj
        try:
            current_pos = get_current_posx()[0]
            cur_x, cur_y, cur_z, cur_Rx, cur_Ry, cur_Rz = current_pos

            # 접근 방향 계산
            dx, dy = x - cur_x, y - cur_y
            if abs(dx) > abs(dy):
                approach_axis = "x+" if dx > 0 else "x-"
            else:
                approach_axis = "y+" if dy > 0 else "y-"

            # 방향별 Rz 보정값
            if approach_axis == "x+":
                Rz_target = 180.0
            elif approach_axis == "x-":
                Rz_target = 0.0
            elif approach_axis == "y+":
                Rz_target = -90.0
            elif approach_axis == "y-":
                Rz_target = 90.0
            
            self.get_logger().error(f"Rz_target : {Rz_target}")

            # 최종 자세 계산
            target_up = [x, y, z + 50, cur_Rx, cur_Ry, Rz_target]
            target_at = [x, y, z,      cur_Rx, cur_Ry, Rz_target]
            home_pose = posj(0, 0, 90, 0, 90, 0)

            # 홈 자세로 이동
            movej(home_pose, VELOCITY, ACC)
            wait(1.0)

            # 그리퍼 열기 (놓기)
            self.gripper.move(0)
            wait(1.5)

            # 단계별 이동
            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            self.get_logger().info(f"→ 접근 ({approach_axis}) 방향으로 이동: {target_at}")
            movel(posx(target_at), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 닫기 (집기)
            self.gripper.move(700)
            wait(1.5)

            # 다시 들어올리기
            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 드롭 지점(조금 옆으로 이동)
            drop_offset = 100.0
            if approach_axis in ("x+", "x-"):
                drop_target = [x, y + drop_offset, z + 50, cur_Rx, cur_Ry, Rz_target]
            else:
                drop_target = [x + drop_offset, y, z + 50, cur_Rx, cur_Ry, Rz_target]

            movel(posx(drop_target), vel=VELOCITY, acc=ACC)
            wait(1.5)

            # 그리퍼 열기 (놓기)
            self.gripper.move(0)
            wait(1.5)

            self.get_logger().info("🏠 홈 자세로 복귀합니다.")
            movej(home_pose, VELOCITY, ACC)
            wait(1.0)

        except Exception as e:
            self.get_logger().error(f"로봇 이동 및 그리퍼 제어 중 오류 발생: {e}")

def main(args=None):
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"DSR_ROBOT2 라이브러리를 임포트할 수 없습니다: {e}")
        rclpy.shutdown()
        exit(1)

    robot_controller = RobotControllerNode()

    # 📌 창 크기 조정 코드 유지
    cv2.namedWindow("RealSense Camera", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("RealSense Camera", 640, 480) 
    cv2.setMouseCallback("RealSense Camera", robot_controller.mouse_callback)

    print("카메라 영상에서 원하는 지점을 클릭하세요. 'ESC' 키를 누르면 종료됩니다.")

    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)

            if robot_controller.latest_cv_color is not None:
                display_image = robot_controller.latest_cv_color.copy()
                
                h, w, _ = display_image.shape
                # 중앙점 표시
                cv2.circle(display_image, (w // 2, h // 2), 3, (0, 0, 255), -1)
                
                cv2.imshow("RealSense Camera", display_image)

            # 이미지 수신 여부와 관계없이 cv2.waitKey는 호출되어야 GUI 이벤트 처리가 됩니다.
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    except KeyboardInterrupt:
        print("Ctrl+C로 종료합니다...")
    except Exception as e:
        print(f"실행 중 예외 발생: {e}")

    finally:
        print("프로그램을 종료합니다...")
        robot_controller.terminate_gripper()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        print("종료 완료.")

if __name__ == '__main__':
    main()
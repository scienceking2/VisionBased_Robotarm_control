# VisionBased Robotarm Control

로봇암(Doosan Robotics) 비전 제어 프로젝트입니다. 인텔 리얼센스(Intel RealSense) 카메라를 활용하여 마커를 인식하고 로봇을 움직이는 제어 코드가 포함되어 있습니다.

## 🚀 목표
- 로봇암 제어 알고리즘 구현 및 경력 쌓기
- 종국적으로 휴머노이드 제어 기술 확보를 위한 기초 단계

## 📦 패키지 구조 (Packages)
- `src/realsense`: 비전(카메라/마커) 및 캘리브레이션 관련 코드를 모듈화한 커스텀 ROS 2 파이썬 패키지.
- `src/doosan-robot2`: (Submodule) 두산 로보틱스 공식 ROS 2 패키지 포크. 제어 및 시뮬레이션을 위한 환경 패키지로, 용량 문제로 서브모듈(바로가기) 처리되었습니다.

## 🛠 설치 및 빌드 방법 (Installation)

### 1. 워크스페이스 클론
이 저장소는 서브모듈(`doosan-robot2`)을 포함하고 있으므로 `--recursive` 옵션으로 클론해야 합니다.

```bash
git clone --recursive https://github.com/scienceking2/VisionBased_Robotarm_control.git ros2_rs
cd ros2_rs
```

만약 이미 클론했다면 다음 명령어를 통해 서브모듈을 최신화할 수 있습니다:
```bash
git submodule update --init --recursive
```

### 2. 패키지 빌드
명령어를 통해 커스텀 패키지 및 두산 로봇 패키지를 빌드합니다.
```bash
cd ~/ros2_rs
colcon build --symlink-install
source install/setup.bash
```

## 🎥 실행 방법 (Usage)

RealSense 비전 제어 노드 실행:
```bash
ros2 run realsense test_realsense
```

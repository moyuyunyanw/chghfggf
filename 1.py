import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 创建保存图片的文件夹
if not os.path.exists('face_images'):
    os.makedirs('face_images')

count = 0  # 图片计数

def draw_detection(image, detection):
    # 获取边界框的属性
    box = detection.location_data.relative_bounding_box
    height, width, _ = image.shape
    xmin = int(box.xmin * width)
    ymin = int(box.ymin * height)
    w = int(box.width * width)
    h = int(box.height * height)

    # 在图像上绘制红色的矩形框
    cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 将图像从 BGR 转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)

        # 绘制检测结果
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                draw_detection(image, detection)
                # 保存检测到人脸的图片
                cv2.imwrite(f'face_images/face_{count}.jpg', image)
                count += 1

        # 显示结果
        cv2.imshow('Real-time Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
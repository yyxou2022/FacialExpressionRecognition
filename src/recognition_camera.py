"""
author: Zhou Chen
datetime: 2019/6/20 15:44
desc: 利用摄像头实时检测
"""
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from model import CNN2, CNN3
from utils import index2emotion, cv2_img_add_text
from blazeface import blaze_detect
# ==========
import cv2
import paho.mqtt.client as mqtt
# from src.recognition_camera import predict_expression
import numpy as np
import base64
import json
# MQTT 服务器连接信息
MQTT_BROKER = "192.168.1.124" #localhost=127.0.0.1=192.168.1.124 localhost:1883=192.168.1.124:1883
MQTT_TOPIC = "facial_expression"   #facial_expression

# 连接 MQTT 服务器
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
# ============

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=1, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None)
opt = parser.parse_args()

if opt.source == 1 and opt.video_path is not None:
    filename = opt.video_path
else:
    filename = None


import subprocess

def avi_to_web_mp4(input_file_path):
    '''
    Converts an AVI video to an MP4 video with H.264 encoding.
    
    @param: [in] input_file_path - The full path of the input AVI video file.
    @return: [output] output_file_path - The full path of the output MP4 video file.
    '''
    output_file_path = input_file_path[:-3] + 'mp4'
    cmd = 'ffmpeg -y -i {} -vcodec h264 {}'.format(input_file_path, output_file_path)
    subprocess.call(cmd, shell=True)
    print("converted done...")
    return output_file_path

def find_most_common(data):
    """
    从数据结构中找出出现次数最多的元素。
    
    参数:
    data -- 可以是列表、元组、集合或其他可迭代对象。
    
    返回:
    出现次数最多的元素及其出现次数。
    """
    # 创建一个字典,用于记录每个元素出现的次数
    count = {}
    for item in data:
        if item in count:
            count[item] += 1
        else:
            count[item] = 1
    
    # 找出出现次数最多的元素
    most_common = max(count, key=count.get)
    most_common_count = count[most_common]
    
    return most_common, most_common_count

def load_model():
    """
    加载本地模型
    :return:
    """
    model = CNN3()
    model.load_weights('/home/yy/Desktop/projects/game/FacialExpressionRecognition/models/cnn3_best_weights.h5')
    return model


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images

# fourcc = cv2.VideoWriter_fourcc(*'I420')
# out = cv2.VideoWriter('./1.mp4', fourcc, 20.0, (800, 600))
# out = cv2.VideoWriter('/home/nypyp/code/metahuman-stream/web/video/captured_video.mp4', fourcc, 20.0, (800, 600))
# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# out = cv2.VideoWriter('1.mp4', fourcc, 20.0, (720, 1280))
def predict_expression():
    """
    实时预测
    :return:
    """
    # 参数设置
    model = load_model()

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 255)  # 白字字
    capture = cv2.VideoCapture(0)  # 指定0号摄像头
    if filename:
        capture = cv2.VideoCapture(filename)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 根据文件名后缀使用合适的编码器
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    #fourcc = cv2.VideoWriter_fourcc(*'I240')
    out = cv2.VideoWriter('/home/nypyp/code/metahuman-stream/web/video/face.avi', fourcc, 20.0, (640, 480))  #720, 1280
    # out = cv2.VideoWriter('/home/yy/Desktop/projects/game/FacialExpressionRecognition/assets/icons/face.avi', fourcc, 30, (640, 480))
    emotions = []
    while True:
        _, frame = capture.read()  # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
        if not _:
            break
        frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)                     
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
        # cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')  # 检测人脸           
        # # 利用分类器识别出哪个区域为人脸
        # faces = cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))
        faces = blaze_detect(frame)
        
        # 如果检测到人脸
        if faces is not None and len(faces) > 0:

            for (x, y, w, h) in faces:
                face = frame[y: y + h, x: x + w]  # 脸部图片
                face = frame_gray[y: y + h, x: x + w]  # 脸部图片
                faces = generate_faces(face)
                results = model.predict(faces)
                result_sum = np.sum(results, axis=0).reshape(-1)
                label_index = np.argmax(result_sum, axis=0)
                emotion = index2emotion(label_index)   #为了方便英文发MQTT服务，将标签转为英文的，进里面进行代码更改
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
                frame = cv2_img_add_text(frame, emotion, x+30, y+30, font_color, 20)
                # print(emotion)
                emotions.append(emotion)

                # puttext中文显示问题
                # cv2.putText(frame, emotion, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4)
                # frame = cv2.cvtColor(cv2.resize(frame, (1004, 576)), cv2.COLOR_RGB2BGR)
        # cv2.imshow("expression recognition(press esc to exit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 利用人眼假象
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        key = cv2.waitKey(30)  # 等待30ms，返回ASCII码

        # 如果输入esc则退出循环
        if key == 27:
            break
    avi_to_web_mp4('/home/nypyp/code/metahuman-stream/web/video/face.avi')
# ==================================
    print(emotions)
    resualt_emo, counts = find_most_common(emotions)
    print("========检测结果：=========",resualt_emo, counts,"=======================")
    emotion_data = {
        "type": resualt_emo,
    }
    payload = json.dumps(emotion_data, ensure_ascii=False).encode("utf-8")
    client.publish(MQTT_TOPIC, payload )
#  =================================
    capture.release()  # 释放摄像头
    out.release()
    cv2.destroyAllWindows()  # 销毁窗口


if __name__ == '__main__':
    predict_expression()

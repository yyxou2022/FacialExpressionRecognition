import os
import time
import cv2
import paho.mqtt.client as mqtt
import numpy as np
import base64
import json
from src.model import CNN3
from src.recognition import predict_expression
# MQTT 服务器连接信息
MQTT_BROKER = "8.134.150.174" #localhost=127.0.0.1=192.168.1.124 localhost:1883=192.168.1.124:1883
MQTT_TOPIC = "facial_expression"
MQTT_TOPIC2 = "topic_fall"

# 连接 MQTT 服务器
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

# # 初始化人脸表情检测器
# detector = FER(mtcnn=True)

model = CNN3()
model.load_weights('models/cnn3_best_weights.h5')

# image_folder = '/home/yy/Desktop/projects/game/FacialExpressionRecognition/input'
image_folder = '/home/nypyp/code/metahuman-stream/web/pages/input'

while True:
    # 遍历文件夹中的所有图片
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            
            # 检测图片并发布结果到 MQTT
            emotions, result_possibilitys = predict_expression(image_path, model)
            emotion_data = {
                "type": emotions,
                # "score": result_possibilitys,
                "file": filename,
                "image": filename,
            }

            # fall_data = {
            #     "time": "20201929",
            #     # "score": result_possibilitys,
            #     "type": "fall",
            # }
            client.publish(MQTT_TOPIC, json.dumps(emotion_data))
            # client.publish(MQTT_TOPIC2, json.dumps(fall_data))
            print(f"已发布图片 {filename} 的人脸表情检测结果")
            time.sleep(5)
            
    # 等待 100 秒后再次检测
    print("等待 100 秒...")
    time.sleep(5)
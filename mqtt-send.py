import cv2
import paho.mqtt.client as mqtt
# from src.recognition_camera import predict_expression
import numpy as np
import base64
import json
from src.model import CNN3
from src.recognition import predict_expression
# MQTT 服务器连接信息
MQTT_BROKER = "8.134.150.174" #localhost=127.0.0.1=192.168.1.124 localhost:1883=192.168.1.124:1883
MQTT_TOPIC = "facial_expression"   #facial_expression

# 连接 MQTT 服务器
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

# # 初始化人脸表情检测器
model = CNN3()
model.load_weights('models/cnn3_best_weights.h5')
emotions, result_possibilitys = predict_expression('/home/yy/Desktop/projects/game/FacialExpressionRecognition/input/test/1.mp4', model)
print(emotions, result_possibilitys)
emotion_data = {
        "type": emotions,
        # "score": result_possibilitys,
}
client.publish(MQTT_TOPIC, json.dumps(emotion_data))
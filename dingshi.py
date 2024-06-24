# -*- coding: utf-8 -*-
import oss2
import os
from oss2.credentials import EnvironmentVariableCredentialsProvider
from oss2.auth import Auth
import os
import time
#
#
access_key_id = 'LTAI5tBrYZn96y1Kybg9K5j3'
access_key_secret = 'jUCdPH9rF7wyDGbG1QP5L0YBqDmxDj'
endpoint = 'http://oss-cn-beijing.aliyuncs.com/'
auth = Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, 'nypyp')
bucket.get_object_to_file('face.mp4', '/home/yy/Desktop/projects/game/FacialExpressionRecognition/input/face.mp4')   #第二个参数下载到指定的文件夹

if __name__ == '__main__':
    while True:
        bucket.get_object_to_file('face.mp4', '/home/yy/Desktop/projects/game/FacialExpressionRecognition/input/face.mp4')   #第二个参数下载到指定的文件夹
        os.system('python /home/yy/Desktop/projects/game/FacialExpressionRecognition/src/recognition_camera.py --source 1 --video_path input/face.mp4')
        time.sleep(5)
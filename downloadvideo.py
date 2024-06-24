#! /usr/bin/env python
import paramiko
import datetime
import os
import time
 

hostname = '192.168.1.124'
username = 'yy'
password = 'yy123456'

port = 22

def download(sftp, local_file, remote_file):
    try:
        print('download file start %s ' % datetime.datetime.now())
        sftp.get(remote_file,local_file)
        print('download file success %s ' % datetime.datetime.now())
    except Exception as e:
        print(e)

def upload(sftp, local_file, remote_file):
    try:
        print('upload file start %s ' % datetime.datetime.now())
        sftp.put(local_file,remote_file)
        print('upload file success %s ' % datetime.datetime.now())
    except Exception as e:
        print(e)


if __name__ == '__main__':
    local_dir = r'/home/yy/Desktop/projects/game/FacialExpressionRecognition/input/test/1.avi'
    remote_dir = r'C://Users//16214//Desktop//1.avi'
    # upmp4_local_dir = r'/home/xin/VIBE/output/upload_video/input/input_vibe_result.mp4'
    # upmp4_remote_dir = r'/home/pyp/apache-tomcat-9.0.78/webapps/big-data-screen/public/input_vibe_result.mp4'
    upglb_local_dir = r'/home/yy/Desktop/projects/game/FacialExpressionRecognition/output.avi'
    upglb_remote_dir = r'C://Users//16214//Desktop//1.avi'
    t = paramiko.Transport((hostname, port))
    t.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(t)
    while True:
        download(sftp, local_dir, remote_dir)
        os.system('python src/recognition_camera.py --source 1 --video_path input/test/1.avi')
        # os.system('python demo.py --vid_file ./downloadvideo/input.mp4 --output_folder output/upload_video/')
        # os.system('python lib/utils/fbx_output.py --input output/upload_video/input/vibe_output.pkl --output output/upload_video/input/fbx_output_vibe.glb --fps_source 30 --fps_target 24 --gender male --person_id 1')
        # upload(sftp, upmp4_local_dir, upmp4_remote_dir)
        upload(sftp, upglb_local_dir, upglb_remote_dir)
        time.sleep(30)
    t.close()
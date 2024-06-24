import paho.mqtt.client as mqtt

# MQTT 服务器连接参数
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "topic_fall"

# 当成功连接到 MQTT 服务器时调用的回调函数
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(TOPIC)

# 当从 MQTT 服务器收到消息时调用的回调函数
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()}")

# 创建 MQTT 客户端实例
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# 连接 MQTT 服务器
client.connect(BROKER_HOST, BROKER_PORT, 60)

# 保持客户端运行,等待消息
client.loop_forever()
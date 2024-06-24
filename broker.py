import asyncio
import websockets
import json
import uuid
import paho.mqtt.client as mqtt


# MQTT 消息存储
topics = {}

async def mqtt_broker(websocket, path):
    # 生成客户端 ID
    client_id = str(uuid.uuid4())

    print(f"New client connected: {client_id}")

    try:
        async for message in websocket:
            # 解析 MQTT 消息
            msg = json.loads(message)
            
            if msg["type"] == "publish":
                # 发布消息到主题
                topic = msg["topic"]
                payload = msg["payload"]
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append((client_id, payload))
                print(f"Published message to topic '{topic}': {payload}")
            
            elif msg["type"] == "subscribe":
                # 订阅主题
                topic = msg["topic"]
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append((client_id, websocket))
                print(f"Client {client_id} subscribed to topic '{topic}'")

    finally:
        # 客户端断开连接时清理订阅信息
        for topic, subscribers in topics.items():
            subscribers[:] = [s for s in subscribers if s[0] != client_id]
        print(f"Client {client_id} disconnected")

start_server = websockets.serve(mqtt_broker, "localhost", 8086)

print("MQTT broker started, listening on ws://localhost:8083")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code "+str(rc))
    client.subscribe("my/topic")

def on_message(client, userdata, msg):
    print("Received message: " + msg.topic + " " + str(msg.payload))

# 在您的服务器初始化代码中添加以下内容
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect("localhost", 8086, 60)
mqtt_client.loop_start()
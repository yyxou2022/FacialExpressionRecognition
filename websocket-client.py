import json
import websocket

def on_message(ws, message):
    print("Received message:", message)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws):
    print("WebSocket connection closed")

def on_open(ws):
    # 订阅 topic
    subscribe_msg = {
        "type": "subscribe",
        "topic": "my/topic"
    }
    ws.send(json.dumps(subscribe_msg))

    # 发布消息
    publish_msg = {
        "type": "publish",
        "topic": "my/topic",
        "payload": "Hello, MQTT over WebSocket!"
    }
    ws.send(json.dumps(publish_msg))

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:8085/mqtt",
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
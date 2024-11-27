# video/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept the WebSocket connection
        await self.accept()

    async def receive(self, text_data):
        # Handle received data (not needed for this example)
        pass

    async def send_video_frame(self, frame):
        # Send a frame (base64) to the WebSocket
        await self.send(text_data=json.dumps({
            'frame': frame
        }))

    async def disconnect(self, close_code):
        # Handle WebSocket disconnect
        pass

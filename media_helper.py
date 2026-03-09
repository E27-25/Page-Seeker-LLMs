import json
import base64
import asyncio
import aiohttp
import cv2
import tempfile
import os

def parse_media_urls(dataset_val):
    if not dataset_val or str(dataset_val) == 'nan':
        return []
    try:
        urls = json.loads(dataset_val)
        return urls if isinstance(urls, list) else []
    except json.JSONDecodeError:
        return []

async def fetch_and_encode_image_async(url, session):
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            content = await response.read()
            return base64.b64encode(content).decode('utf-8')
    except Exception as e:
        # Expected if links are broken
        return None

def _extract_frames_sync(temp_path, num_frames):
    frames = []
    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            step = max(1, total_frames // num_frames)
            for i in range(num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 640))
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
    except Exception as e:
        print(f"Error processing video frames: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return frames

async def extract_frames_from_video_async(url, session, num_frames=1):
    try:
        async with session.get(url, timeout=20) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
        
        loop = asyncio.get_running_loop()
        frames = await loop.run_in_executor(None, _extract_frames_sync, tmp_path, num_frames)
        return frames
    except Exception as e:
        return []

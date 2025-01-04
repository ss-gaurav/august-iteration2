import cv2
import threading
import os

class DataLoader:
    def __init__(self, video_path = 'detection-testing/videos/trimmed2.avi'):

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise Exception(f"Cannot open video file: {self.video_path}")
        
    
    def getframe(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release(self):
        self.cap.release



class DataLoaderCsiCamera:
    def __init__(self, pipeline )
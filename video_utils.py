#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from tqdm import tqdm


class VideoFrameExtractor:
    """视频帧提取器，用于高效地从视频中提取帧"""
    
    def __init__(self, video_path):
        """
        初始化视频帧提取器
        
        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def extract_frames_at_interval(self, interval=1.0, roi=None):
        """
        按时间间隔提取视频帧
        
        Args:
            interval: 时间间隔（秒）
            roi: 感兴趣区域，格式为(x, y, width, height)，默认为整帧
            
        Returns:
            frames: 提取的帧列表
            timestamps: 对应的时间戳列表
        """
        frames = []
        timestamps = []
        
        frame_interval = int(self.fps * interval)
        frame_interval = max(1, frame_interval)  # 确保至少提取一帧
        
        total_frames_to_extract = int(self.frame_count / frame_interval) + 1
        
        with tqdm(total=total_frames_to_extract, desc="提取视频帧") as pbar:
            current_frame = 0
            while current_frame < self.frame_count:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                timestamp = current_frame / self.fps
                
                # 如果指定了ROI，则只提取感兴趣区域
                if roi:
                    x, y, w, h = roi
                    frame = frame[y:y+h, x:x+w]
                
                frames.append(frame)
                timestamps.append(timestamp)
                pbar.update(1)
                
                current_frame += frame_interval
        
        return frames, timestamps


class FrameProcessor:
    """视频帧处理器，用于对提取的帧进行预处理"""
    
    @staticmethod
    def enhance_subtitle(frame):
        """
        增强字幕区域，提高OCR识别率
        
        Args:
            frame: 输入帧
            
        Returns:
            增强后的帧
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用自适应阈值二值化，提高文本对比度
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 增强边缘
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated
    
    @staticmethod
    def remove_noise(frame):
        """
        去除噪点
        
        Args:
            frame: 输入帧
            
        Returns:
            去噪后的帧
        """
        # 应用高斯模糊去除噪点
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 应用中值滤波进一步去除噪点
        denoised = cv2.medianBlur(blurred, 3)
        
        return denoised
    
    @staticmethod
    def detect_text_regions(frame):
        """
        检测帧中可能包含文本的区域
        
        Args:
            frame: 输入帧
            
        Returns:
            包含文本区域的矩形列表，格式为[(x, y, w, h), ...]
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能包含文本的区域
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 根据形状特征筛选可能的文本区域
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 20.0 and w > 40 and h > 8:
                text_regions.append((x, y, w, h))
        
        return text_regions 
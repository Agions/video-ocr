#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频OCR字幕提取工具示例脚本

使用方法：
    python example.py 你的视频文件路径
"""

import os
import sys
from main import VideoOCR


def run_example():
    if len(sys.argv) < 2:
        print("使用方法: python example.py 视频文件路径")
        return 1
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return 1
    
    # 设置输出文件路径
    output_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_subtitles.srt")
    
    print(f"正在处理视频: {video_path}")
    print(f"输出字幕文件: {output_path}")
    
    # 使用默认参数处理视频
    ocr = VideoOCR(
        video_path,
        subtitle_area=(0.7, 1.0),  # 识别画面下方30%区域
        lang='zh',                 # 中文识别
        interval=1.0               # 每1秒提取一帧
    )
    
    # 处理视频并生成字幕
    ocr.process_video(
        output_path,
        min_duration=0.3,          # 最小字幕持续时间
        enhance_frames=True,       # 增强帧以提高OCR识别率
        debug=False                # 不输出调试信息
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(run_example()) 
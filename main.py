#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
from datetime import timedelta
from tqdm import tqdm
import cv2
import numpy as np

# 尝试导入可选依赖
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("警告: 未找到scikit-image模块，将使用替代方法计算相似度")
    SKIMAGE_AVAILABLE = False

# 尝试导入pysrt
try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    print("警告: 未找到pysrt模块，将使用内置方法生成SRT文件")
    PYSRT_AVAILABLE = False

# 导入自定义模块
from video_utils import VideoFrameExtractor, FrameProcessor
from ocr_utils import OCREngine, SubtitleFilter

# 检查必要的依赖库是否安装
def check_dependencies():
    """检查必要的依赖库是否安装"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pytesseract
    except ImportError:
        missing_deps.append("pytesseract")
    
    if missing_deps:
        print(f"错误: 缺少必要的依赖库: {', '.join(missing_deps)}")
        print("请运行: pip install " + " ".join(missing_deps))
        sys.exit(1)

# 初始化时检查依赖
check_dependencies()


class VideoOCR:
    """视频OCR字幕提取工具"""
    
    def __init__(self, video_path, subtitle_area=(0.7, 1.0), lang='chi_sim', interval=1.0):
        """
        初始化视频OCR工具
        
        Args:
            video_path: 视频文件路径
            subtitle_area: 字幕区域范围，格式为(开始高度比例, 结束高度比例)
            lang: 识别语言，Tesseract语言代码，例如'chi_sim'(简体中文)、'eng'(英文)
            interval: 提取帧的时间间隔（秒）
        """
        self.video_path = video_path
        self.subtitle_area = subtitle_area
        self.lang = lang
        self.interval = interval
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 初始化视频提取器
        self.extractor = VideoFrameExtractor(video_path)
        
        # 计算字幕区域
        height = self.extractor.height
        width = self.extractor.width
        y_start = int(height * subtitle_area[0])
        y_end = int(height * subtitle_area[1])
        self.subtitle_roi = (0, y_start, width, y_end - y_start)
        
        # 初始化OCR引擎
        self.ocr = OCREngine(lang=lang)
        
        # 创建字幕过滤器
        self.subtitle_filter = SubtitleFilter()
        
        # 输出视频信息
        print(f"视频信息：")
        print(f"  - 文件: {os.path.basename(video_path)}")
        print(f"  - 分辨率: {width}x{height}")
        print(f"  - 帧率: {self.extractor.fps:.2f} fps")
        print(f"  - 总帧数: {self.extractor.frame_count}")
        print(f"  - 时长: {timedelta(seconds=int(self.extractor.duration))}")
        print(f"  - 字幕区域: 高度 {subtitle_area[0]*100:.0f}%-{subtitle_area[1]*100:.0f}%")
    
    def extract_frames(self):
        """
        提取视频帧
        
        Returns:
            元组 (frames, timestamps)，其中frames是帧图像列表，timestamps是对应的时间戳
        """
        # 计算采样间隔
        interval_frames = int(self.interval * self.extractor.fps)
        if interval_frames < 1:
            interval_frames = 1
            print(f"警告: 指定的间隔过小，调整为最小间隔 (1/{self.extractor.fps:.2f} 秒)")
        
        # 提取帧
        frames = []
        timestamps = []
        last_frame = None
        last_similarity = 0
        
        # 显示进度条
        total_frames = self.extractor.frame_count
        frame_indices = range(0, total_frames, interval_frames)
        
        print(f"提取帧间隔: {interval_frames} 帧 ({self.interval:.2f} 秒)")
        print(f"预计提取帧数: {len(frame_indices)}")
        
        for i in tqdm(frame_indices, desc="提取视频帧"):
            success, frame = self.extractor.get_frame_at_index(i)
            if not success:
                continue
            
            # 提取字幕区域
            x, y, w, h = self.subtitle_roi
            subtitle_region = frame[y:y+h, x:x+w]
            
            # 场景变化检测，跳过相似的帧
            if last_frame is not None:
                # 比较当前帧与上一帧的相似度
                similarity = FrameProcessor.compute_similarity(subtitle_region, last_frame)
                
                # 如果相似度高，则跳过此帧
                if similarity > 0.98:
                    # 但如果已经跳过很多相似帧，强制保留一帧以防止长时间没有字幕
                    if last_similarity > 0.98 and len(frames) > 0 and (i - timestamps[-1] * self.extractor.fps) > 5 * interval_frames:
                        pass  # 不跳过
                    else:
                        last_similarity = similarity
                        continue
            
            # 保存当前帧用于下一次比较
            last_frame = subtitle_region.copy()
            last_similarity = 0 if last_frame is None else last_similarity
            
            # 添加帧和时间戳
            frames.append(subtitle_region)
            timestamp = i / self.extractor.fps
            timestamps.append(timestamp)
        
        print(f"实际提取帧数: {len(frames)}")
        return frames, timestamps
    
    def extract_subtitles(self, output_file):
        """
        从视频中提取字幕并保存为SRT文件
        
        Args:
            output_file: 输出SRT文件路径
            
        Returns:
            成功提取的字幕数量
        """
        start_time = time.time()
        
        # 提取帧
        print("\n1. 开始提取视频帧...")
        frames, timestamps = self.extract_frames()
        
        if not frames:
            raise ValueError("未能提取到任何视频帧")
        
        # 识别文本
        print("\n2. 开始OCR识别...")
        texts = self.ocr.batch_recognize(frames)
        
        # 清理和过滤结果
        print("\n3. 处理识别结果...")
        filtered_texts = []
        filtered_timestamps = []
        
        for i, text in enumerate(texts):
            # 过滤空白和过短文本
            filtered = self.subtitle_filter.filter_by_length(text, min_length=1)
            if filtered:
                # 进行文本后处理
                processed = self.subtitle_filter.post_process_text(filtered)
                filtered_texts.append(processed)
                filtered_timestamps.append(timestamps[i])
        
        print(f"  - 初步识别字幕数: {len(filtered_texts)}")
        
        # 去除重复
        unique_texts, unique_timestamps = self.subtitle_filter.remove_duplicates(
            filtered_texts, 
            [(t, t + self.interval) for t in filtered_timestamps]
        )
        
        print(f"  - 去重后字幕数: {len(unique_texts)}")
        
        # 合并相邻相同字幕
        merged_texts, merged_timestamps = self.subtitle_filter.merge_adjacent(
            unique_texts, 
            unique_timestamps,
            max_gap=1.5  # 稍微增大间隔，更好地合并分散的字幕
        )
        
        print(f"  - 合并后字幕数: {len(merged_texts)}")
        
        # 生成SRT文件
        print(f"\n4. 生成SRT文件: {output_file}")
        srt_generator = SRTGenerator()
        srt_generator.generate(merged_texts, merged_timestamps, output_file)
        
        # 计算总耗时
        end_time = time.time()
        process_time = end_time - start_time
        
        print(f"\n处理完成!")
        print(f"  - 总耗时: {process_time:.1f}秒")
        print(f"  - 最终字幕数: {len(merged_texts)}")
        print(f"  - 输出文件: {output_file}")
        
        return len(merged_texts)
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'extractor'):
            self.extractor.release()


class VideoFrameExtractor:
    """视频帧提取工具"""
    
    def __init__(self, video_path):
        """
        初始化视频帧提取器
        
        Args:
            video_path: 视频文件路径
        """
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def get_frame_at_index(self, frame_index):
        """
        获取指定索引的帧
        
        Args:
            frame_index: 帧索引
            
        Returns:
            元组 (success, frame)
        """
        if frame_index >= self.frame_count:
            return False, None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return self.cap.read()
    
    def get_frame_at_time(self, time_seconds):
        """
        获取指定时间的帧
        
        Args:
            time_seconds: 时间(秒)
            
        Returns:
            元组 (success, frame)
        """
        frame_index = int(time_seconds * self.fps)
        return self.get_frame_at_index(frame_index)
    
    def release(self):
        """释放资源"""
        if self.cap.isOpened():
            self.cap.release()


class FrameProcessor:
    """帧处理工具，用于增强OCR效果"""
    
    @staticmethod
    def preprocess_for_ocr(image):
        """
        预处理图像以提高OCR识别效果
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用自适应阈值处理，增强字幕文本
        # 字幕通常是白色文字，可以考虑进行反色处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 反转回来（如果字幕是白色文字黑色背景）
        binary = cv2.bitwise_not(binary)
        
        # 降噪
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    @staticmethod
    def compute_similarity(frame1, frame2):
        """
        计算两帧之间的相似度
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            
        Returns:
            相似度值，范围0-1，1表示完全相同
        """
        # 确保两帧尺寸相同
        if frame1.shape != frame2.shape:
            # 调整大小以匹配
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # 转为灰度图
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2
        
        # 计算结构相似性
        if SKIMAGE_AVAILABLE:
            try:
                similarity = ssim(gray1, gray2)
                return max(0, min(1, similarity))  # 确保值在0-1范围内
            except Exception:
                # 如果SSIM计算失败，使用备用方法
                pass
                
        # 使用直方图比较作为备用方法
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # 归一化直方图
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # 计算相似度
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, min(1, similarity))  # 确保值在0-1范围内


class SRTGenerator:
    """SRT字幕生成器"""
    
    def generate(self, texts, timestamps, output_file):
        """
        生成SRT字幕文件
        
        Args:
            texts: 字幕文本列表
            timestamps: 时间戳列表，格式为[(start_time, end_time), ...]
            output_file: 输出SRT文件路径
            
        Returns:
            生成的字幕数量
        """
        if not texts or not timestamps:
            return 0
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 如果pysrt可用，使用pysrt创建SRT文件
        if PYSRT_AVAILABLE:
            try:
                subs = pysrt.SubRipFile()
                
                for i, (text, (start_time, end_time)) in enumerate(zip(texts, timestamps)):
                    # 创建SubRipItem
                    sub = pysrt.SubRipItem(
                        index=i+1, 
                        start=self._seconds_to_time(start_time), 
                        end=self._seconds_to_time(end_time), 
                        text=text
                    )
                    subs.append(sub)
                
                # 保存SRT文件
                subs.save(output_file, encoding='utf-8')
                return len(texts)
            except Exception as e:
                print(f"使用pysrt创建SRT文件时发生错误: {e}")
                print("使用内置方法生成SRT文件...")
        
        # 如果pysrt不可用或出错，使用内置方法创建SRT文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (text, (start_time, end_time)) in enumerate(zip(texts, timestamps)):
                # 格式化时间戳
                start_time_str = self._format_timestamp(start_time)
                end_time_str = self._format_timestamp(end_time)
                
                # 写入SRT格式
                f.write(f"{i+1}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{text}\n\n")
        
        return len(texts)
    
    def _seconds_to_time(self, seconds):
        """
        将秒转换为pysrt时间对象
        
        Args:
            seconds: 秒数
            
        Returns:
            pysrt时间对象
        """
        if not PYSRT_AVAILABLE:
            return None
            
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return pysrt.SubRipTime(hours, minutes, int(seconds), milliseconds)
    
    def _format_timestamp(self, seconds):
        """
        将秒数格式化为SRT时间戳格式 HH:MM:SS,mmm
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间戳字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def main():
    """主函数入口点"""
    parser = argparse.ArgumentParser(description='视频OCR字幕提取工具')
    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('-o', '--output', help='输出SRT文件路径')
    parser.add_argument('--area', default='0.7,1.0', help='字幕区域，格式为"开始高度,结束高度"，范围0.0-1.0')
    parser.add_argument('--lang', default='chi_sim', help='识别语言，例如chi_sim(简体中文)、eng(英文)')
    parser.add_argument('--interval', type=float, default=1.0, help='提取帧的时间间隔(秒)')
    
    args = parser.parse_args()
    
    try:
        # 解析字幕区域
        area_parts = args.area.split(',')
        subtitle_area = (float(area_parts[0]), float(area_parts[1]))
        
        # 设置默认输出文件
        output_file = args.output
        if not output_file:
            base_name = os.path.splitext(os.path.basename(args.video))[0]
            output_file = f"{base_name}.srt"
        
        # 创建OCR提取器
        ocr = VideoOCR(
            args.video,
            subtitle_area=subtitle_area,
            lang=args.lang,
            interval=args.interval
        )
        
        # 提取字幕
        try:
            ocr.extract_subtitles(output_file)
            print(f"字幕已保存至: {output_file}")
        finally:
            ocr.close()
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
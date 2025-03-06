#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class OCREngine:
    """OCR引擎，封装Tesseract的功能"""
    
    def __init__(self, lang='chi_sim', config='--psm 6'):
        """
        初始化OCR引擎
        
        Args:
            lang: 识别语言，可选值为chi_sim（简体中文）、eng（英文）、jpn（日语）等
                  完整列表可通过tesseract --list-langs查看
            config: Tesseract配置参数
        """
        self.lang = lang
        self.config = config
        
        # 验证Tesseract是否可用
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract版本: {version}")
            
            # 检查lang是否可用
            langs = pytesseract.get_languages()
            if self.lang not in langs and not '+' in self.lang:
                main_lang = self.lang.split('+')[0] if '+' in self.lang else self.lang
                if main_lang not in langs:
                    print(f"警告: 语言包 {self.lang} 不可用，默认使用eng")
                    print(f"可用语言包: {', '.join(langs)}")
                    print("请安装所需语言包: ")
                    print("  - macOS: brew install tesseract-lang")
                    print("  - Ubuntu: sudo apt-get install tesseract-ocr-<lang>")
                    self.lang = "eng"
        except Exception as e:
            print(f"警告: Tesseract可能未正确安装: {str(e)}")
            print("请确保系统中安装了Tesseract OCR引擎:")
            print("  - macOS: brew install tesseract tesseract-lang")
            print("  - Ubuntu: sudo apt-get install tesseract-ocr")
            print("  - Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装")
    
    def recognize_text(self, image):
        """
        识别图像中的文本
        
        Args:
            image: 输入图像
            
        Returns:
            识别出的文本字符串
        """
        try:
            # 预处理图像以提高OCR质量
            processed_image = self._preprocess_image(image)
            
            # 进行OCR识别
            text = pytesseract.image_to_string(processed_image, lang=self.lang, config=self.config)
            
            # 清理文本
            text = text.strip()
            
            return text
        except Exception as e:
            print(f"OCR识别错误: {str(e)}")
            return ""
    
    def _preprocess_image(self, image):
        """
        预处理图像以提高OCR识别效果
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 转为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 调整对比度
        # 这对字幕识别特别有用，因为字幕通常是高对比度的
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 去噪处理
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # 二值化处理 - 使用自适应阈值
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 字幕通常是白色文字，所以反转图像（可根据需要调整）
        # 取决于视频字幕的颜色
        # 如果是黑底白字，则不需要反转
        # 如果是白底黑字，则需要反转
        # 这里可以添加检测逻辑，根据图像直方图决定是否反转
        mean_val = cv2.mean(gray)[0]
        if mean_val > 127:  # 如果平均亮度较高，可能是白底黑字
            binary = cv2.bitwise_not(binary)
        
        # 膨胀操作可以帮助连接断开的文字部分
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 返回处理后的图像
        return dilated
    
    def batch_recognize(self, images, batch_size=10):
        """
        批量识别多张图像中的文本
        
        Args:
            images: 图像列表
            batch_size: 每批处理的图像数量
            
        Returns:
            识别出的文本列表
        """
        if not images:
            return []
            
        results = []
        
        # 预处理所有图像
        processed_images = [self._preprocess_image(img) for img in images]
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            # 定义处理函数
            def process_image(img):
                return pytesseract.image_to_string(img, lang=self.lang, config=self.config).strip()
            
            # 并行处理图像
            results = list(tqdm(
                executor.map(process_image, processed_images),
                total=len(processed_images),
                desc="识别文本"
            ))
        
        return results


class SubtitleFilter:
    """字幕过滤器，用于过滤和清理OCR识别结果"""
    
    @staticmethod
    def filter_by_length(text, min_length=1, max_length=100):
        """
        根据文本长度过滤
        
        Args:
            text: 输入文本
            min_length: 最小长度
            max_length: 最大长度
            
        Returns:
            过滤后的文本，不符合条件则返回空字符串
        """
        if min_length <= len(text) <= max_length:
            # 去除空白字符和无意义的字符
            text = text.strip()
            if text:
                return text
        return ""
    
    @staticmethod
    def remove_duplicates(texts, timestamps):
        """
        去除重复的字幕
        
        Args:
            texts: 文本列表
            timestamps: 时间戳列表，格式为[(start_time, end_time), ...]
            
        Returns:
            去重后的文本列表和对应的时间戳列表
        """
        if not texts:
            return [], []
        
        filtered_texts = [texts[0]]
        filtered_timestamps = [timestamps[0]]
        
        for i in range(1, len(texts)):
            # 如果当前文本与前一个不同，则保留
            if texts[i] != filtered_texts[-1]:
                filtered_texts.append(texts[i])
                filtered_timestamps.append(timestamps[i])
        
        return filtered_texts, filtered_timestamps
    
    @staticmethod
    def merge_adjacent(texts, timestamps, max_gap=1.0):
        """
        合并相邻的相同字幕
        
        Args:
            texts: 文本列表
            timestamps: 时间戳列表，格式为[(start_time, end_time), ...]
            max_gap: 允许合并的最大时间间隔（秒）
            
        Returns:
            合并后的文本列表和对应的时间戳列表
        """
        if not texts:
            return [], []
        
        merged_texts = [texts[0]]
        merged_timestamps = [timestamps[0]]
        
        for i in range(1, len(texts)):
            current_text = texts[i]
            current_start, current_end = timestamps[i]
            prev_start, prev_end = merged_timestamps[-1]
            
            # 如果文本相同且时间间隔小于阈值，则合并
            if (current_text == merged_texts[-1] and 
                current_start - prev_end <= max_gap):
                # 更新结束时间
                merged_timestamps[-1] = (prev_start, current_end)
            else:
                merged_texts.append(current_text)
                merged_timestamps.append(timestamps[i])
        
        return merged_texts, merged_timestamps
    
    @staticmethod
    def post_process_text(text):
        """
        对识别出的文本进行后处理
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
            
        # 去除多余的空白字符
        processed = ' '.join(text.split())
        
        # 去除常见的OCR错误和干扰字符
        replacements = {
            '|': 'I',
            '0': 'O',
            '1': 'I',
            'l': 'I',
            '「': '"',
            '」': '"',
            '『': '"',
            '』': '"',
            '…': '...',
        }
        
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        # 删除重复的标点符号
        punctuations = ['.', ',', '!', '?', ';', ':', '"']
        for punct in punctuations:
            processed = processed.replace(punct + punct, punct)
        
        # 去除无用字符
        chars_to_remove = ['~', '`', '@', '#', '$', '%', '^', '*', '_', '{', '}', '<', '>', '\\', '/', '=', '+']
        for char in chars_to_remove:
            processed = processed.replace(char, '')
        
        # 最后再次清理多余空白
        processed = ' '.join(processed.split())
        
        return processed 
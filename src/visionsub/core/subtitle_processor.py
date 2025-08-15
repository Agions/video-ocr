"""
Subtitle Processor - Handles OCR result processing and subtitle generation
"""
import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.ocr_engine import OCRResult
from ..models.config import ProcessingConfig
from ..models.subtitle import SubtitleItem

logger = logging.getLogger(__name__)


@dataclass
class ProcessingContext:
    """处理上下文信息"""
    video_duration: float
    fps: float
    frame_count: int
    video_width: int
    video_height: int


class TextCleaner:
    """文本清理器"""

    def __init__(self, config: ProcessingConfig):
        self.config = config

        # 常见的OCR错误映射
        self.error_mappings = {
            # 数字和字母混淆
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '2': 'Z',
            '8': 'B',
            # 标点符号修复
            ',,': ',',
            '..': '.',
            '。。': '。',
            '，，': '，',
            '!!': '!',
            '??': '?',
            # 空格处理
            '  ': ' ',
            ' \n': '\n',
            '\n ': '\n',
        }

        # 语言特定的清理规则
        self.language_rules = {
            "中文": {
                "remove_english": False,
                "fix_punctuation": True,
                "merge_spaces": True,
            },
            "英文": {
                "remove_chinese": True,
                "fix_punctuation": True,
                "merge_spaces": True,
            }
        }

    def clean_text(self, text: str, language: str = "中文") -> str:
        """
        清理和增强文本

        Args:
            text: 原始文本
            language: 文本语言

        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""

        cleaned = text

        # 应用语言特定规则
        rules = self.language_rules.get(language, {})

        # 移除不需要的字符
        if rules.get("remove_english", False) and language == "英文":
            cleaned = re.sub(r'[\u4e00-\u9fff]', '', cleaned)  # 移除中文

        if rules.get("remove_chinese", False) and language == "中文":
            cleaned = re.sub(r'[a-zA-Z]', '', cleaned)  # 移除英文

        # 修复常见的OCR错误
        for wrong, correct in self.error_mappings.items():
            cleaned = cleaned.replace(wrong, correct)

        # 修复标点符号
        if rules.get("fix_punctuation", True):
            cleaned = self._fix_punctuation(cleaned, language)

        # 合并空格
        if rules.get("merge_spaces", True):
            cleaned = ' '.join(cleaned.split())

        # 移除特殊字符
        cleaned = self._remove_special_characters(cleaned, language)

        return cleaned.strip()

    def _fix_punctuation(self, text: str, language: str) -> str:
        """修复标点符号"""
        if language == "中文":
            # 中文标点符号修复
            text = re.sub(r'([a-zA-Z0-9]),', r'\1，', text)
            text = re.sub(r'([a-zA-Z0-9])\.', r'\1。', text)
            text = re.sub(r'([a-zA-Z0-9])!', r'\1！', text)
            text = re.sub(r'([a-zA-Z0-9])\?', r'\1？', text)
        else:
            # 英文标点符号修复
            text = re.sub(r'([a-zA-Z0-9])，', r'\1,', text)
            text = re.sub(r'([a-zA-Z0-9])。', r'\1.', text)
            text = re.sub(r'([a-zA-Z0-9])！', r'\1!', text)
            text = re.sub(r'([a-zA-Z0-9])？', r'\1?', text)

        return text

    def _remove_special_characters(self, text: str, language: str) -> str:
        """移除特殊字符"""
        # 保留字母、数字、基本标点和常用符号
        if language == "中文":
            # 保留中文字符、英文、数字和基本标点
            pattern = r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s,.!?;:()\[\]{}"\'-]'
        else:
            # 保留英文、数字和基本标点
            pattern = r'[^a-zA-Z0-9\s,.!?;:()\[\]{}"\'-]'

        return re.sub(pattern, '', text)

    def validate_text(self, text: str) -> bool:
        """验证文本是否有效"""
        if not text or len(text.strip()) == 0:
            return False

        # 检查是否包含有效字符
        if re.match(r'^[\s\W]*$', text):
            return False

        # 检查长度
        if len(text.strip()) > 500:  # 限制字幕长度
            return False

        return True


class TimelineOptimizer:
    """时间轴优化器"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.min_duration = timedelta(seconds=0.5)  # 最小显示时间
        self.max_duration = timedelta(seconds=10.0)  # 最大显示时间
        self.merge_threshold = timedelta(seconds=1.0)  # 合并阈值

    def optimize_timeline(self, items: List[SubtitleItem]) -> List[SubtitleItem]:
        """
        优化字幕时间轴

        Args:
            items: 原始字幕项目列表

        Returns:
            List[SubtitleItem]: 优化后的字幕项目列表
        """
        if not items:
            return []

        # 按开始时间排序
        sorted_items = sorted(items, key=lambda x: x.start_time)

        # 优化时间轴
        optimized = []
        current_group = [sorted_items[0]]

        for item in sorted_items[1:]:
            last_item = current_group[-1]

            # 检查是否可以合并
            if self._can_merge(last_item, item):
                current_group.append(item)
            else:
                # 处理当前组
                optimized.extend(self._process_group(current_group))
                current_group = [item]

        # 处理最后一组
        if current_group:
            optimized.extend(self._process_group(current_group))

        return optimized

    def _can_merge(self, item1: SubtitleItem, item2: SubtitleItem) -> bool:
        """检查两个字幕是否可以合并"""
        # 检查时间间隔
        time_gap = item2.start_time - item1.end_time
        if time_gap > self.merge_threshold:
            return False

        # 检查位置相似性（如果有位置信息）
        if hasattr(item1, 'position') and hasattr(item2, 'position'):
            pos1 = item1.position
            pos2 = item2.position
            distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            if distance > 100:  # 位置差异太大
                return False

        return True

    def _process_group(self, group: List[SubtitleItem]) -> List[SubtitleItem]:
        """处理一组字幕"""
        if not group:
            return []

        if len(group) == 1:
            return [self._adjust_single_item(group[0])]

        # 合并多个字幕
        merged_text = ' '.join(item.text for item in group)

        # 计算时间范围
        start_time = min(item.start_time for item in group)
        end_time = max(item.end_time for item in group)

        # 确保最小显示时间
        if end_time - start_time < self.min_duration:
            end_time = start_time + self.min_duration

        # 限制最大显示时间
        if end_time - start_time > self.max_duration:
            end_time = start_time + self.max_duration

        # 计算平均置信度
        avg_confidence = np.mean([item.confidence for item in group])

        return [SubtitleItem(
            index=group[0].index,
            start_time=start_time,
            end_time=end_time,
            text=merged_text,
            confidence=avg_confidence,
            language=group[0].language,
            metadata={
                "merged_count": len(group),
                "original_indices": [item.index for item in group]
            }
        )]

    def _adjust_single_item(self, item: SubtitleItem) -> SubtitleItem:
        """调整单个字幕项目"""
        start_time = item.start_time
        end_time = item.end_time

        # 确保最小显示时间
        if end_time - start_time < self.min_duration:
            end_time = start_time + self.min_duration

        # 限制最大显示时间
        if end_time - start_time > self.max_duration:
            end_time = start_time + self.max_duration

        return SubtitleItem(
            index=item.index,
            start_time=start_time,
            end_time=end_time,
            text=item.text,
            confidence=item.confidence,
            language=item.language,
            metadata=item.metadata
        )


class SubtitleProcessor:
    """字幕处理器"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.text_cleaner = TextCleaner(config)
        self.timeline_optimizer = TimelineOptimizer(config)

        # 时间轴计算参数
        self.default_duration = timedelta(seconds=2.0)
        self.time_threshold = timedelta(seconds=0.1)

    async def process_ocr_results(self, ocr_results: List[Tuple[OCRResult, float]],
                                 context: ProcessingContext) -> List[SubtitleItem]:
        """
        处理OCR结果生成字幕

        Args:
            ocr_results: OCR结果和时间戳列表 [(OCRResult, timestamp)]
            context: 视频处理上下文

        Returns:
            List[SubtitleItem]: 字幕项目列表
        """
        if not ocr_results:
            return []

        logger.info(f"Processing {len(ocr_results)} OCR results")

        # 转换OCR结果为字幕项目
        raw_subtitles = []
        index = 1

        for ocr_result, timestamp in ocr_results:
            frame_subtitles = self._ocr_result_to_subtitles(ocr_result, timestamp, index)
            raw_subtitles.extend(frame_subtitles)
            index += len(frame_subtitles)

        # 清理和验证字幕
        cleaned_subtitles = []
        for subtitle in raw_subtitles:
            cleaned_text = self.text_cleaner.clean_text(subtitle.text, subtitle.language)

            if self.text_cleaner.validate_text(cleaned_text):
                subtitle.text = cleaned_text
                cleaned_subtitles.append(subtitle)

        logger.info(f"Generated {len(cleaned_subtitles)} subtitle items after cleaning")

        # 优化时间轴
        optimized_subtitles = self.timeline_optimizer.optimize_timeline(cleaned_subtitles)

        logger.info(f"Optimized to {len(optimized_subtitles)} subtitle items")

        return optimized_subtitles

    def _ocr_result_to_subtitles(self, ocr_result: OCRResult, timestamp: float,
                                start_index: int) -> List[SubtitleItem]:
        """
        将OCR结果转换为字幕项目

        Args:
            ocr_result: OCR结果
            timestamp: 时间戳
            start_index: 起始索引

        Returns:
            List[SubtitleItem]: 字幕项目列表
        """
        subtitles = []

        for i, item in enumerate(ocr_result.items):
            # 计算时间范围
            start_time = timedelta(seconds=timestamp)
            end_time = start_time + self.default_duration

            # 创建字幕项目
            subtitle = SubtitleItem(
                index=start_index + i,
                start_time=start_time,
                end_time=end_time,
                text=item.text,
                confidence=item.confidence,
                language=item.language or self.config.ocr_config.language,
                metadata={
                    "source_frame": timestamp,
                    "bbox": item.bbox,
                    "ocr_engine": ocr_result.engine_info.get("engine", "unknown"),
                    "processing_time": ocr_result.processing_time
                }
            )
            subtitles.append(subtitle)

        return subtitles

    async def merge_adjacent_subtitles(self, subtitles: List[SubtitleItem]) -> List[SubtitleItem]:
        """
        合并相邻的字幕

        Args:
            subtitles: 字幕列表

        Returns:
            List[SubtitleItem]: 合并后的字幕列表
        """
        if not subtitles:
            return []

        # 按开始时间排序
        sorted_subs = sorted(subtitles, key=lambda x: x.start_time)

        merged = []
        current_group = [sorted_subs[0]]

        for subtitle in sorted_subs[1:]:
            last_subtitle = current_group[-1]

            # 检查是否可以合并
            if self._should_merge_subtitles(last_subtitle, subtitle):
                current_group.append(subtitle)
            else:
                # 处理当前组
                merged.append(self._merge_subtitle_group(current_group))
                current_group = [subtitle]

        # 处理最后一组
        if current_group:
            merged.append(self._merge_subtitle_group(current_group))

        return merged

    def _should_merge_subtitles(self, sub1: SubtitleItem, sub2: SubtitleItem) -> bool:
        """判断两个字幕是否应该合并"""
        # 检查时间间隔
        time_gap = sub2.start_time - sub1.end_time
        if time_gap > self.time_threshold:
            return False

        # 检查文本长度（避免合并过长的文本）
        combined_length = len(sub1.text) + len(sub2.text)
        if combined_length > 100:  # 限制合并后的长度
            return False

        return True

    def _merge_subtitle_group(self, group: List[SubtitleItem]) -> SubtitleItem:
        """合并字幕组"""
        if len(group) == 1:
            return group[0]

        # 合并文本
        merged_text = ' '.join(sub.text for sub in group)

        # 计算时间范围
        start_time = min(sub.start_time for sub in group)
        end_time = max(sub.end_time for sub in group)

        # 计算平均置信度
        avg_confidence = np.mean([sub.confidence for sub in group])

        return SubtitleItem(
            index=group[0].index,
            start_time=start_time,
            end_time=end_time,
            text=merged_text,
            confidence=avg_confidence,
            language=group[0].language,
            metadata={
                "merged_count": len(group),
                "original_indices": [sub.index for sub in group]
            }
        )

    async def filter_subtitles(self, subtitles: List[SubtitleItem],
                             min_confidence: float = None,
                             min_length: int = None,
                             max_length: int = None) -> List[SubtitleItem]:
        """
        过滤字幕

        Args:
            subtitles: 字幕列表
            min_confidence: 最小置信度
            min_length: 最小长度
            max_length: 最大长度

        Returns:
            List[SubtitleItem]: 过滤后的字幕列表
        """
        if min_confidence is None:
            min_confidence = self.config.ocr_config.confidence_threshold

        if min_length is None:
            min_length = 1

        if max_length is None:
            max_length = 200

        filtered = []
        for subtitle in subtitles:
            # 检查置信度
            if subtitle.confidence < min_confidence:
                continue

            # 检查长度
            text_length = len(subtitle.text)
            if text_length < min_length or text_length > max_length:
                continue

            filtered.append(subtitle)

        return filtered

    def validate_subtitle(self, subtitle: SubtitleItem) -> bool:
        """
        验证字幕项目

        Args:
            subtitle: 字幕项目

        Returns:
            bool: 是否有效
        """
        # 检查时间
        if subtitle.start_time >= subtitle.end_time:
            return False

        # 检查文本
        if not self.text_cleaner.validate_text(subtitle.text):
            return False

        # 检查置信度
        if subtitle.confidence < 0:
            return False

        return True

    def get_statistics(self, subtitles: List[SubtitleItem]) -> Dict[str, Any]:
        """
        获取字幕统计信息

        Args:
            subtitles: 字幕列表

        Returns:
            Dict[str, Any]: 统计信息
        """
        if not subtitles:
            return {}

        total_subtitles = len(subtitles)
        avg_confidence = np.mean([sub.confidence for sub in subtitles])
        avg_length = np.mean([len(sub.text) for sub in subtitles])

        # 语言分布
        languages = {}
        for sub in subtitles:
            lang = sub.language or "unknown"
            languages[lang] = languages.get(lang, 0) + 1

        # 时间分布
        durations = [(sub.end_time - sub.start_time).total_seconds() for sub in subtitles]
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        min_duration = np.min(durations)

        return {
            "total_subtitles": total_subtitles,
            "average_confidence": float(avg_confidence),
            "average_length": float(avg_length),
            "language_distribution": languages,
            "duration_stats": {
                "average": float(avg_duration),
                "max": float(max_duration),
                "min": float(min_duration)
            },
            "time_range": {
                "start": str(min(sub.start_time for sub in subtitles)),
                "end": str(max(sub.end_time for sub in subtitles))
            }
        }

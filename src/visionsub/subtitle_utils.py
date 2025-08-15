#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta

import pysrt


class SRTGenerator:
    """SRT字幕生成器"""

    @staticmethod
    def create_srt(texts, timestamps, output_path):
        """
        创建SRT字幕文件

        Args:
            texts: 字幕文本列表
            timestamps: 时间戳列表，格式为[(start_time, end_time), ...]
            output_path: 输出文件路径
        """
        subs = pysrt.SubRipFile()

        for i, (text, (start_time, end_time)) in enumerate(zip(texts, timestamps)):
            if not text.strip():
                continue

            item = pysrt.SubRipItem()
            item.index = i + 1

            # 设置开始时间
            start_seconds = int(start_time)
            start_millis = int((start_time - start_seconds) * 1000)

            item.start.hours = start_seconds // 3600
            item.start.minutes = (start_seconds % 3600) // 60
            item.start.seconds = start_seconds % 60
            item.start.milliseconds = start_millis

            # 设置结束时间
            end_seconds = int(end_time)
            end_millis = int((end_time - end_seconds) * 1000)

            item.end.hours = end_seconds // 3600
            item.end.minutes = (end_seconds % 3600) // 60
            item.end.seconds = end_seconds % 60
            item.end.milliseconds = end_millis

            # 设置文本
            item.text = text

            subs.append(item)

        # 保存SRT文件
        subs.save(output_path, encoding='utf-8')

        return len(subs)

    @staticmethod
    def format_timedelta(seconds):
        """
        将秒数格式化为 HH:MM:SS,mmm 格式

        Args:
            seconds: 时间（秒）

        Returns:
            格式化后的时间字符串
        """
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @staticmethod
    def merge_srt_files(srt_files, output_path):
        """
        合并多个SRT文件

        Args:
            srt_files: SRT文件路径列表
            output_path: 输出文件路径
        """
        merged = pysrt.SubRipFile()
        current_index = 1

        for srt_file in srt_files:
            subs = pysrt.open(srt_file, encoding='utf-8')

            for sub in subs:
                sub.index = current_index
                merged.append(sub)
                current_index += 1

        merged.save(output_path, encoding='utf-8')


class SubtitleAdjuster:
    """字幕调整器，用于调整字幕的时间和内容"""

    @staticmethod
    def shift_time(srt_file, seconds, output_path=None):
        """
        调整字幕时间

        Args:
            srt_file: SRT文件路径
            seconds: 调整的秒数，正数为延后，负数为提前
            output_path: 输出文件路径，默认覆盖原文件
        """
        subs = pysrt.open(srt_file, encoding='utf-8')

        for sub in subs:
            sub.shift(seconds=seconds)

        if output_path:
            subs.save(output_path, encoding='utf-8')
        else:
            subs.save(srt_file, encoding='utf-8')

    @staticmethod
    def adjust_duration(srt_file, factor, output_path=None):
        """
        调整字幕持续时间

        Args:
            srt_file: SRT文件路径
            factor: 调整因子，大于1表示延长，小于1表示缩短
            output_path: 输出文件路径，默认覆盖原文件
        """
        subs = pysrt.open(srt_file, encoding='utf-8')

        for sub in subs:
            # 计算中点时间
            midpoint = sub.start.ordinal + (sub.end.ordinal - sub.start.ordinal) / 2

            # 计算新的持续时间
            duration = sub.end.ordinal - sub.start.ordinal
            new_duration = duration * factor

            # 设置新的开始和结束时间
            new_start = midpoint - new_duration / 2
            new_end = midpoint + new_duration / 2

            # 将新的开始和结束时间应用到字幕
            sub.start = pysrt.SubRipTime.from_ordinal(int(new_start))
            sub.end = pysrt.SubRipTime.from_ordinal(int(new_end))

        if output_path:
            subs.save(output_path, encoding='utf-8')
        else:
            subs.save(srt_file, encoding='utf-8')

    @staticmethod
    def merge_short_subtitles(srt_file, min_duration=1.0, max_gap=0.5, output_path=None):
        """
        合并短时间字幕

        Args:
            srt_file: SRT文件路径
            min_duration: 最小字幕持续时间（秒）
            max_gap: 允许合并的最大时间间隔（秒）
            output_path: 输出文件路径，默认覆盖原文件
        """
        subs = pysrt.open(srt_file, encoding='utf-8')

        i = 0
        while i < len(subs) - 1:
            current = subs[i]
            next_sub = subs[i + 1]

            # 计算当前字幕的持续时间（毫秒）
            current_duration = current.end.ordinal - current.start.ordinal

            # 计算与下一个字幕的间隔（毫秒）
            gap = next_sub.start.ordinal - current.end.ordinal

            # 如果当前字幕持续时间小于最小持续时间且与下一个字幕间隔小于最大间隔
            if (current_duration < min_duration * 1000 and gap < max_gap * 1000):
                # 合并字幕
                current.end = next_sub.end
                current.text += "\n" + next_sub.text

                # 删除下一个字幕
                subs.remove(next_sub)
            else:
                i += 1

        # 重新编号
        for i, sub in enumerate(subs):
            sub.index = i + 1

        if output_path:
            subs.save(output_path, encoding='utf-8')
        else:
            subs.save(srt_file, encoding='utf-8')

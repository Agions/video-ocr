# 视频OCR字幕提取工具

一个高效的视频硬字幕提取工具，可以从视频中提取字幕并输出为SRT格式文件。参考了[SubtitleOCR](https://github.com/nhjydywd/SubtitleOCR)的设计理念，使用Python实现，并兼容最新版本Mac系统。

## 功能特点

- 支持多种视频格式的字幕提取
- 高效的字幕区域检测
- 使用PaddleOCR进行高精度文字识别
- 支持输出为SRT格式字幕文件
- 多线程并行处理，提高处理速度

## 环境要求

- Python 3.8+
- macOS 10.15+（兼容最新版macOS）
- 所需依赖包见`requirements.txt`

## 安装方法

1. 克隆本仓库
```bash
git clone https://github.com/yourusername/video-ocr-tool.git
cd video-ocr-tool
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：
```bash
python main.py --video 视频路径 --output 输出路径
```

更多选项：
```bash
python main.py --video 视频路径 --output 输出路径 --subtitle_area 0.7,1.0 --lang zh --interval 1
```

参数说明：
- `--video`：输入视频文件路径
- `--output`：输出SRT文件路径
- `--subtitle_area`：字幕区域范围，格式为"开始高度比例,结束高度比例"，默认为0.7,1.0，表示识别画面下方30%区域
- `--lang`：识别语言，默认为zh（中文）
- `--interval`：视频帧提取间隔（秒），默认为1秒

## 许可证

MIT
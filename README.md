# VisionSub - 专业视频OCR字幕提取工具

VisionSub 是一款现代化的功能丰富的视频OCR应用程序，具有先进的文本识别、实时预览和多格式导出功能，可从视频中提取字幕。

## 🚀 新功能与改进

### 核心架构增强
- **统一处理引擎**: 具有模块化架构的清晰关注点分离
- **场景变化检测**: 智能帧处理，跳过相似帧，性能提升70%以上
- **高级缓存系统**: 具有自适应策略的LRU缓存，用于重复OCR操作
- **全面错误处理**: 强大的错误恢复，提供用户友好的消息和故障排除建议

### 智能ROI选择系统
- **交互式ROI选择**: 鼠标拖拽选择感兴趣区域，实时可视化反馈
- **多ROI管理**: 支持创建、编辑、删除多个ROI区域
- **ROI预设系统**: 内置常见字幕区域预设（底部、顶部、左侧、右侧、中心等）
- **ROI配置管理**: 每个ROI可独立配置语言、置信度阈值等参数
- **ROI序列化**: 支持保存和加载ROI配置，便于重复使用
- **智能ROI应用**: 自动处理ROI边界验证，确保OCR处理的准确性

### 高级OCR功能
- **文本增强**: 自动纠正常见OCR错误和标点符号修复
- **语言自动检测**: 多语言内容的智能语言检测
- **图像预处理**: 去噪、对比度增强、阈值处理和锐化
- **置信度评分**: 具有视觉反馈的实时置信度指示器

### 实时预览和编辑
- **实时OCR预览**: 具有叠加可视化的实时文本提取
- **交互式ROI选择**: 拖放感兴趣区域选择，支持多ROI管理
- **高级ROI管理**: ROI预设、启用/禁用、自定义配置和序列化
- **字幕编辑器**: 具有撤销/重做功能的完整编辑功能
- **视觉置信度显示**: 彩色编码的置信度指示器和边界框

### 多种导出格式
- **SRT (SubRip Text)**: 标准字幕格式
- **WebVTT**: Web视频文本轨道格式
- **ASS (Advanced SubStation Alpha)**: 具有样式的高级字幕格式
- **纯文本**: 简单文本导出
- **JSON**: 用于进一步处理的结构化数据导出
- **批量导出**: 同时导出为多种格式

### 性能优化
- **并行处理**: 多线程帧处理
- **自适应帧采样**: 基于视频持续时间的智能帧间隔计算
- **内存管理**: 高效的资源清理和内存优化
- **处理统计**: 实时性能指标和优化洞察

## 系统要求

- Python 3.9+
- 现代OCR引擎（推荐PaddleOCR）
- 依赖项:
  - PyQt6 (现代GUI框架)
  - OpenCV (计算机视觉)
  - NumPy (数值计算)
  - PaddleOCR (高级OCR引擎)
  - Pydantic (数据验证)
  - pysrt (字幕处理)
  - tqdm (进度条)

## 安装

### 1. 安装Python依赖

```bash
# 使用Poetry（推荐）
poetry install

# 或使用pip
pip install -e .
```

### 2. 安装OCR引擎

#### PaddleOCR（推荐）
```bash
# 安装PaddlePaddle
pip install paddlepaddle

# 安装PaddleOCR
pip install paddleocr
```

#### 备选OCR引擎
应用程序通过插件系统支持多种OCR引擎。

## 快速开始

### 基本用法

```python
from visionsub import ProcessingEngine, ProcessingConfig

# 创建配置
config = ProcessingConfig(
    ocr_config=OCRConfig(
        engine="PaddleOCR",
        language="ch",
        auto_detect_language=True,
        roi_rect=(0, 300, 640, 100)  # 底部字幕区域
    ),
    scene_threshold=0.3
)

# 创建处理引擎
engine = ProcessingEngine(config)

# 处理视频
subtitles = await engine.process_video("input.mp4")

# 导出为多种格式
from visionsub.export import ExportManager
export_manager = ExportManager()
await export_manager.export_multiple_formats(
    subtitles, 
    "output", 
    ['.srt', '.vtt', '.json']
)

# ROI管理示例
roi_manager = engine.get_roi_manager()

# 添加自定义ROI
roi_id = roi_manager.add_roi(
    name="自定义字幕区域",
    roi_type="custom",
    rect=(50, 400, 540, 80),
    description="用户自定义的字幕识别区域"
)

# 设置活动ROI
roi_manager.set_active_roi(roi_id)

# 保存ROI配置
roi_manager.save_rois("my_roi_config.json")

# 使用预设ROI
presets = [
    {"name": "顶部标题", "type": "title", "rect": [0, 0, 640, 60]},
    {"name": "底部字幕", "type": "subtitle", "rect": [0, 420, 640, 60]}
]
roi_manager.import_roi_presets(presets)
```

### GUI应用程序

启动图形界面：

```bash
# 使用Poetry
poetry run python -m visionsub.ui.main

# 或使用已安装的包
visionsub-gui
```

### 命令行界面

```bash
# 基本用法
visionsub-cli input.mp4 -o output.srt

# 带选项的高级用法
visionsub-cli input.mp4 \
    --language ch \
    --scene-threshold 0.3 \
    --cache-size 100 \
    --export-formats srt,vtt,json \
    --output-dir ./exports
```

## 架构概述

### 核心组件

```
visionsub/
├── core/                    # 核心处理引擎
│   ├── engine.py          # 主要处理协调器
│   ├── video_processor.py # 优化的视频处理
│   ├── scene_detection.py # 场景变化检测
│   ├── frame_cache.py     # 智能缓存系统
│   ├── roi_manager.py     # ROI管理和应用
│   └── errors.py          # 全面错误处理
├── features/              # 高级功能
│   ├── advanced_ocr.py    # 增强的OCR和文本改进
│   └── preview.py         # 实时预览和编辑
├── export/                # 导出功能
│   └── export_manager.py  # 多格式导出系统
├── models/                # 数据模型
│   ├── config.py          # 配置模型
│   ├── subtitle.py        # 字幕数据模型
│   └── video.py           # 视频元数据模型
└── ui/                    # 用户界面
    ├── main_window.py     # 主窗口
    ├── video_player.py    # 视频播放器
    └── roi_selection.py   # ROI选择面板
```

### 关键设计模式

- **清洁架构**: 具有依赖注入的关注点分离
- **MVVM模式**: UI组件的模型-视图-视图模型
- **策略模式**: 可插拔的OCR引擎和导出格式
- **观察者模式**: 组件之间的事件驱动通信
- **工厂模式**: 处理组件的动态创建

## 开发

### 运行测试

```bash
# 运行所有测试
poetry run pytest

# 运行覆盖率测试
poetry run pytest --cov=visionsub --cov-report=html

# 运行特定测试类别
poetry run pytest -m "not slow"        # 仅快速测试
poetry run pytest -m "integration"     # 集成测试
poetry run pytest -m "slow"            # 性能测试
```

### 代码质量

```bash
# 格式化代码
poetry run ruff format

# 检查代码
poetry run ruff check

# 类型检查
poetry run mypy visionsub
```

### 性能优化

应用程序包含多项性能优化：

- **场景变化检测**: 通过跳过相似帧减少70%以上的处理
- **自适应缓存**: 具有可配置大小和过期时间的LRU缓存
- **并行处理**: 大型视频的多线程帧处理
- **内存管理**: 自动资源清理和内存优化

## API参考

### ProcessingEngine

```python
engine = ProcessingEngine(config)
subtitles = await engine.process_video("video.mp4")
result = await engine.process_frame(frame, timestamp)
stats = engine.get_processing_stats()
```

### ExportManager

```python
export_manager = ExportManager()
await export_manager.export_subtitles(subtitles, "output.srt")
files = await export_manager.export_multiple_formats(
    subtitles, "base", ['.srt', '.vtt', '.json']
)
```

### AdvancedOCRProcessor

```python
ocr = AdvancedOCRProcessor(config)
result = await ocr.process_with_enhancement(frame, timestamp)
```

## 配置

### OCR配置

```python
OCRConfig(
    engine="PaddleOCR",
    language="ch",
    denoise=True,
    enhance_contrast=True,
    threshold=128,
    sharpen=True,
    auto_detect_language=True,
    roi_rect=(0, 300, 640, 100)  # ROI区域 (x, y, width, height)
)
```

### 处理配置

```python
ProcessingConfig(
    ocr_config=ocr_config,
    scene_threshold=0.3,
    cache_size=100
)
```

## 故障排除

### 常见问题

**性能问题**
- 启用场景变化检测以获得更好性能
- 根据可用内存调整缓存大小
- 为视频长度使用适当的帧间隔

**OCR准确性**
- 确保字幕区域的适当ROI选择
- 使用图像预处理功能（去噪、增强对比度）
- 尝试不同的OCR引擎以获得更好结果

**内存使用**
- 监控处理过程中的内存使用情况
- 为大型视频调整缓存大小
- 在处理运行之间启用垃圾回收

### 错误消息

应用程序提供详细的错误消息和故障排除建议：

- **视频处理错误**: 检查视频格式和完整性
- **OCR错误**: 验证OCR引擎安装和语言包
- **导出错误**: 确保写入权限和磁盘空间

## 贡献

1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 为新功能添加测试
5. 确保所有测试通过
6. 提交拉取请求

### 开发设置

```bash
# 克隆仓库
git clone <repository-url>
cd visionsub

# 安装开发依赖
poetry install --with dev

# 运行pre-commit钩子
poetry run pre-commit install
```

## 许可证

MIT许可证 - 详情请参见LICENSE文件

## 更新日志

### 版本1.0.0
- 具有现代架构的完全重写
- 场景变化检测，性能提升70%以上
- 具有文本增强功能的高级OCR功能
- 实时预览和编辑功能
- 多种导出格式支持
- 全面的错误处理和恢复
- 具有集成测试的完整测试覆盖
- 现代化的PyQt6 GUI界面
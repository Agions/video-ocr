# VisionSub API 参考文档

## 概述

VisionSub 提供了完整的API接口，支持开发者进行二次开发和集成。本文档详细介绍了所有可用的API接口、数据结构和使用示例。

## 目录

1. [核心模块](#核心模块)
2. [UI组件](#ui组件)
3. [数据处理](#数据处理)
4. [配置管理](#配置管理)
5. [扩展接口](#扩展接口)
6. [事件系统](#事件系统)
7. [错误处理](#错误处理)
8. [最佳实践](#最佳实践)

## 核心模块

### MainViewModel

主视图模型，管理应用程序的核心业务逻辑。

#### 类定义
```python
class MainViewModel(QObject):
    """主视图模型，管理应用程序状态和业务逻辑"""
    
    # 信号定义
    frame_changed = pyqtSignal(np.ndarray)           # 帧变化信号
    config_changed = pyqtSignal(OcrConfig)           # 配置变化信号
    single_ocr_result_changed = pyqtSignal(str)      # OCR结果变化信号
    error_occurred = pyqtSignal(str)                 # 错误信号
    video_loaded = pyqtSignal(int)                   # 视频加载完成信号
    is_playing_changed = pyqtSignal(bool)            # 播放状态变化信号
    frame_index_changed = pyqtSignal(int)            # 帧索引变化信号
    roi_config_changed = pyqtSignal(dict)            # ROI配置变化信号
    subtitles_processed = pyqtSignal(list)          # 字幕处理完成信号
    queue_changed = pyqtSignal(list)                 # 队列变化信号
    batch_progress_changed = pyqtSignal(int)         # 批处理进度信号
    batch_status_changed = pyqtSignal(str)            # 批处理状态信号
```

#### 主要方法

##### `__init__()`
```python
def __init__(self):
    """初始化主视图模型"""
    super().__init__()
    
    # 状态管理
    self.config = OcrConfig()
    self.video_reader: Optional[VideoReader] = None
    self.current_frame: Optional[np.ndarray] = None
    self.current_frame_index = 0
    self.is_playing = False
    self.timer = QTimer()
    
    # 服务组件
    self.batch_service = BatchService()
    self.processing_engine = ProcessingEngine(ProcessingConfig())
    self.roi_manager = self.processing_engine.get_roi_manager()
```

##### `load_video(file_path: str) -> bool`
```python
def load_video(self, file_path: str) -> bool:
    """
    加载视频文件
    
    Args:
        file_path (str): 视频文件路径
        
    Returns:
        bool: 加载是否成功
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持
        RuntimeError: 视频读取失败
    """
    try:
        self.video_reader = VideoReader(file_path)
        self.current_frame_index = 0
        self.current_frame = self.video_reader.get_frame(0)
        self.frame_changed.emit(self.current_frame)
        self.video_loaded.emit(self.video_reader.metadata.frame_count)
        return True
    except Exception as e:
        self.error_occurred.emit(f"Failed to load video: {e}")
        return False
```

##### `play()` / `pause()` / `stop()`
```python
def play(self):
    """开始视频播放"""
    if self.video_reader and not self.is_playing:
        self.is_playing = True
        self.timer.start(33)  # ~30 FPS
        self.is_playing_changed.emit(True)

def pause(self):
    """暂停视频播放"""
    if self.is_playing:
        self.is_playing = False
        self.timer.stop()
        self.is_playing_changed.emit(False)

def stop(self):
    """停止视频播放并重置到开始"""
    self.pause()
    if self.video_reader:
        self.current_frame_index = 0
        self.current_frame = self.video_reader.get_frame(0)
        self.frame_changed.emit(self.current_frame)
        self.frame_index_changed.emit(0)
```

##### `update_config(config: OcrConfig)`
```python
def update_config(self, config: OcrConfig):
    """
    更新OCR配置
    
    Args:
        config (OcrConfig): 新的OCR配置
    """
    self.config = config
    self.config_changed.emit(config)
    # 更新OCR引擎
    self.ocr_engine = get_ocr_engine(self.config.engine)
```

##### `update_roi_config(roi_config: Dict[str, Any])`
```python
def update_roi_config(self, roi_config: Dict[str, Any]):
    """
    更新ROI配置
    
    Args:
        roi_config (dict): ROI配置字典
    """
    self.processing_engine.set_roi_config(roi_config)
    
    if "roi_rect" in roi_config:
        self.config.roi_rect = roi_config["roi_rect"]
        self.config_changed.emit(self.config)
    
    self.roi_config_changed.emit(roi_config)
```

##### `async run_single_frame_ocr()`
```python
async def run_single_frame_ocr(self):
    """
    对当前帧执行OCR识别
    
    Returns:
        None (结果通过single_ocr_result_changed信号发送)
    """
    if self.current_frame is not None:
        try:
            result = await self.processing_engine.process_frame(
                self.current_frame,
                timestamp=self.current_frame_index / 30.0,
                apply_roi=True
            )
            
            # 格式化OCR结果
            if result.get('roi_applied', False):
                roi_info = result.get('roi_info', {})
                roi_name = roi_info.get('name', 'Unknown')
                text = f"[ROI: {roi_name}] " + " ".join(result.get('text', []))
            else:
                text = " ".join(result.get('text', []))
            
            # 添加置信度信息
            confidence = result.get('confidence', 0.0)
            if confidence > 0:
                text += f" (置信度: {confidence:.2f})"
            
            self.single_ocr_result_changed.emit(text)
        except Exception as e:
            self.error_occurred.emit(f"OCR failed: {e}")
```

### ProcessingEngine

处理引擎，负责视频处理和OCR识别的核心逻辑。

#### 类定义
```python
class ProcessingEngine:
    """视频处理引擎，协调各个处理组件"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.roi_manager = ROIManager()
        self.ocr_engine = get_ocr_engine(config.ocr_config.engine)
        self.video_processor = VideoProcessor()
        self.scene_detector = SceneDetector(config.scene_threshold)
        self.frame_cache = FrameCache(config.cache_size)
```

#### 主要方法

##### `async process_frame(frame: np.ndarray, timestamp: float, apply_roi: bool = True) -> Dict[str, Any]`
```python
async def process_frame(
    self, 
    frame: np.ndarray, 
    timestamp: float, 
    apply_roi: bool = True
) -> Dict[str, Any]:
    """
    处理单帧图像
    
    Args:
        frame (np.ndarray): 输入帧
        timestamp (float): 时间戳
        apply_roi (bool): 是否应用ROI
        
    Returns:
        dict: 处理结果字典
    """
    result = {
        'timestamp': timestamp,
        'frame_shape': frame.shape,
        'text': [],
        'confidence': 0.0,
        'roi_applied': False,
        'roi_info': None
    }
    
    try:
        # 应用ROI
        if apply_roi and self.roi_manager.has_active_roi():
            roi_rect = self.roi_manager.get_active_roi_rect()
            if roi_rect and self._is_valid_roi(frame.shape, roi_rect):
                frame = self._apply_roi(frame, roi_rect)
                result['roi_applied'] = True
                result['roi_info'] = self.roi_manager.get_active_roi_info()
        
        # 执行OCR
        ocr_result = await self.ocr_engine.recognize(frame)
        
        # 后处理
        processed_text = self._post_process_text(ocr_result['text'])
        
        result.update({
            'text': processed_text,
            'confidence': ocr_result['confidence'],
            'language': ocr_result.get('language', 'unknown')
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Frame processing failed: {e}")
        return result
```

##### `async process_video(video_path: str) -> List[Subtitle]`
```python
async def process_video(self, video_path: str) -> List[Subtitle]:
    """
    处理整个视频
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        List[Subtitle]: 字幕列表
    """
    try:
        video_reader = VideoReader(video_path)
        subtitles = []
        
        # 场景检测
        scene_changes = await self.scene_detector.detect_scenes(video_reader)
        
        # 处理每个场景
        for i, (start_frame, end_frame) in enumerate(scene_changes):
            scene_subtitles = await self._process_scene(
                video_reader, start_frame, end_frame
            )
            subtitles.extend(scene_subtitles)
        
        return subtitles
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise
```

##### `set_roi_config(roi_config: Dict[str, Any])`
```python
def set_roi_config(self, roi_config: Dict[str, Any]):
    """
    设置ROI配置
    
    Args:
        roi_config (dict): ROI配置字典
    """
    self.roi_manager.update_config(roi_config)
```

### ROIManager

ROI管理器，处理区域选择和管理。

#### 类定义
```python
class ROIManager:
    """ROI管理器，处理区域选择和管理"""
    
    roi_added = pyqtSignal(ROIInfo)
    roi_removed = pyqtSignal(str)
    roi_updated = pyqtSignal(ROIInfo)
    active_roi_changed = pyqtSignal(ROIInfo)
```

#### 主要方法

##### `add_roi(name: str, roi_type: ROIType, rect: Tuple[int, int, int, int], **kwargs) -> str`
```python
def add_roi(
    self, 
    name: str, 
    roi_type: ROIType, 
    rect: Tuple[int, int, int, int], 
    **kwargs
) -> str:
    """
    添加新的ROI
    
    Args:
        name (str): ROI名称
        roi_type (ROIType): ROI类型
        rect (tuple): ROI矩形区域 (x, y, width, height)
        **kwargs: 其他参数
        
    Returns:
        str: ROI ID
    """
    roi_id = str(uuid.uuid4())
    roi = ROIInfo(
        id=roi_id,
        name=name,
        type=roi_type,
        rect=rect,
        enabled=kwargs.get('enabled', True),
        confidence_threshold=kwargs.get('confidence_threshold', 0.0),
        language=kwargs.get('language', ''),
        description=kwargs.get('description', '')
    )
    
    self.rois[roi_id] = roi
    self.roi_added.emit(roi)
    
    return roi_id
```

##### `remove_roi(roi_id: str)`
```python
def remove_roi(self, roi_id: str):
    """
    删除ROI
    
    Args:
        roi_id (str): ROI ID
    """
    if roi_id in self.rois:
        del self.rois[roi_id]
        self.roi_removed.emit(roi_id)
        
        # 如果删除的是活动ROI，清除活动状态
        if self.active_roi_id == roi_id:
            self.active_roi_id = None
```

##### `set_active_roi(roi_id: str)`
```python
def set_active_roi(self, roi_id: str):
    """
    设置活动ROI
    
    Args:
        roi_id (str): ROI ID
    """
    if roi_id in self.rois:
        self.active_roi_id = roi_id
        self.active_roi_changed.emit(self.rois[roi_id])
```

## UI组件

### EnhancedMainWindow

增强主窗口，应用程序的主要界面。

#### 类定义
```python
class EnhancedMainWindow(QMainWindow):
    """增强主窗口，具有现代化UI/UX设计和安全功能"""
```

#### 主要方法

##### `__init__(view_model: MainViewModel)`
```python
def __init__(self, view_model: MainViewModel):
    """
    初始化主窗口
    
    Args:
        view_model (MainViewModel): 视图模型实例
    """
    super().__init__()
    self.vm = view_model
    self.theme_manager = get_theme_manager()
    
    self.setWindowTitle("VisionSub - 视频OCR字幕提取工具")
    self.setGeometry(100, 100, 1600, 900)
    self.setMinimumSize(1200, 700)
    
    self.setup_ui()
    self.connect_signals()
    self.setup_theme()
```

##### `setup_ui()`
```python
def setup_ui(self):
    """设置UI界面"""
    # 创建中央窗口部件
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    
    # 主布局
    main_layout = QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)
    
    # 创建工具栏
    self.create_toolbar()
    
    # 创建主分割器
    self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
    main_layout.addWidget(self.main_splitter)
    
    # 创建左侧面板（视频播放器）
    self.video_player = EnhancedVideoPlayer(self.vm)
    self.main_splitter.addWidget(self.video_player)
    
    # 创建右侧面板（OCR预览和ROI管理）
    right_panel = self.create_right_panel()
    self.main_splitter.addWidget(right_panel)
    
    # 设置分割器比例
    self.main_splitter.setSizes([800, 400])
    
    # 创建状态栏
    self.create_status_bar()
```

### EnhancedVideoPlayer

增强视频播放器，支持ROI选择和视频控制。

#### 类定义
```python
class EnhancedVideoPlayer(StyledWidget):
    """增强视频播放器，具有现代化控件和安全功能"""
    
    frame_updated = pyqtSignal(np.ndarray)
    roi_selected = pyqtSignal(QRect)
    playback_state_changed = pyqtSignal(bool)
```

#### 主要方法

##### `update_frame(frame: np.ndarray)`
```python
def update_frame(self, frame: np.ndarray):
    """
    更新显示的视频帧
    
    Args:
        frame (np.ndarray): 视频帧
    """
    try:
        # 安全验证
        if frame is None:
            logger.warning("接收到空帧")
            return
            
        # 检查帧尺寸
        if hasattr(frame, 'shape'):
            height, width = frame.shape[:2]
            if width > self._max_frame_width or height > self._max_frame_height:
                logger.error(f"帧尺寸超过安全限制: {width}x{height}")
                return
        
        self.current_frame = frame
        self.display_frame(frame)
        self.frame_updated.emit(frame)
        
    except Exception as e:
        logger.error(f"帧更新失败: {e}")
```

##### `set_roi(roi: QRect)`
```python
def set_roi(self, roi: QRect):
    """
    设置ROI区域
    
    Args:
        roi (QRect): ROI矩形区域
    """
    self.current_roi = roi
    self.roi_overlay.set_current_roi(roi)
    self.roi_selected.emit(roi)
```

### EnhancedOCRPreview

增强OCR预览组件，显示OCR结果和提供交互功能。

#### 类定义
```python
class EnhancedOCRPreview(StyledWidget):
    """增强OCR预览组件，具有安全渲染和现代设计"""
    
    result_selected = pyqtSignal(OCRResult)
    result_edited = pyqtSignal(OCRResult)
    text_exported = pyqtSignal(str)
    filter_applied = pyqtSignal(float, float)
```

#### 主要方法

##### `add_result(result: OCRResult)`
```python
def add_result(self, result: OCRResult):
    """
    添加OCR结果
    
    Args:
        result (OCRResult): OCR结果对象
    """
    sanitized_result = self.renderer.sanitize_result(result)
    self.results.append(sanitized_result)
    self.text_editor.append_result(sanitized_result)
    self.apply_filter()
    self.update_statistics()
```

##### `set_results(results: List[OCRResult])`
```python
def set_results(self, results: List[OCRResult]):
    """
    设置OCR结果列表
    
    Args:
        results (List[OCRResult]): OCR结果列表
    """
    self.results = self.renderer.sanitize_results(results)
    self.text_editor.clear_results()
    
    for result in self.results:
        self.text_editor.append_result(result)
    
    self.apply_filter()
    self.update_statistics()
```

##### `apply_filter()`
```python
def apply_filter(self):
    """
    应用置信度过滤器
    """
    min_confidence, max_confidence = self.confidence_filter.get_range()
    
    self.filtered_results = [
        result for result in self.results
        if min_confidence <= result.confidence <= max_confidence
    ]
    
    self.results_table.set_results(self.filtered_results)
    self.filter_applied.emit(min_confidence, max_confidence)
    self.update_statistics()
```

## 数据处理

### OCRResult

OCR结果数据结构。

#### 类定义
```python
@dataclass
class OCRResult:
    """OCR结果数据结构"""
    text: str
    confidence: float
    language: str
    position: QRect
    timestamp: float
    result_type: OCRResultType = OCRResultType.RAW
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
```

#### 使用示例
```python
# 创建OCR结果
result = OCRResult(
    text="这是识别的文本",
    confidence=0.95,
    language="zh",
    position=QRect(10, 10, 200, 30),
    timestamp=1.0,
    result_type=OCRResultType.PROCESSED,
    metadata={"source": "frame_001"}
)

# 添加到预览组件
ocr_preview.add_result(result)
```

### Subtitle

字幕数据结构。

#### 类定义
```python
@dataclass
class Subtitle:
    """字幕数据结构"""
    index: int
    start_time: float
    end_time: float
    text: str
    language: str = "zh"
    confidence: float = 0.0
    position: Optional[QRect] = None
    style: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.style is None:
            self.style = {}
```

#### 使用示例
```python
# 创建字幕
subtitle = Subtitle(
    index=1,
    start_time=1.0,
    end_time=3.0,
    text="这是字幕文本",
    language="zh",
    confidence=0.95,
    position=QRect(10, 100, 400, 30),
    style={"color": "#FFFFFF", "font-size": "16px"}
)

# 导出为SRT格式
srt_content = subtitle.to_srt()
```

## 配置管理

### AppConfig

应用程序配置。

#### 类定义
```python
class AppConfig(BaseModel):
    """完整的应用程序配置"""
    
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    app_name: str = Field(default="VisionSub")
    version: str = Field(default="2.0.0")
    config_file_path: Optional[str] = Field(default=None)
```

#### 主要方法

##### `load_from_file(config_path: str) -> AppConfig`
```python
@classmethod
def load_from_file(cls, config_path: str) -> 'AppConfig':
    """
    从文件加载配置
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        AppConfig: 配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误
    """
    import json
    from pathlib import Path
    
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    return cls(**data)
```

##### `save_to_file(config_path: str)`
```python
def save_to_file(self, config_path: str):
    """
    保存配置到文件
    
    Args:
        config_path (str): 配置文件路径
    """
    import json
    from pathlib import Path
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = self.model_dump()
    
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

### OcrConfig

OCR配置。

#### 类定义
```python
class OcrConfig(BaseModel):
    """OCR配置"""
    
    engine: Literal["PaddleOCR", "Tesseract"] = Field(default="PaddleOCR")
    language: str = Field(default="中文")
    threshold: int = Field(default=180, ge=0, le=255)
    roi_rect: Tuple[int, int, int, int] = Field(default=(0, 0, 0, 0))
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_preprocessing: bool = Field(default=True)
    enable_postprocessing: bool = Field(default=True)
    custom_params: Optional[dict] = Field(default=None)
```

## 扩展接口

### OCR引擎接口

#### 基础接口
```python
class OCREngine(ABC):
    """OCR引擎基础接口"""
    
    @abstractmethod
    async def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        识别图像中的文本
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            dict: 识别结果
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str):
        """
        设置识别语言
        
        Args:
            language (str): 语言代码
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表
        
        Returns:
            List[str]: 语言列表
        """
        pass
```

#### 实现示例
```python
class PaddleOCREngine(OCREngine):
    """PaddleOCR引擎实现"""
    
    def __init__(self, language: str = "ch"):
        self.language = language
        self.ocr = PaddleOCR(use_angle_cls=True, lang=language)
    
    async def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        result = self.ocr.ocr(image, cls=True)
        
        return {
            'text': [line[1][0] for line in result[0]],
            'confidence': sum(line[1][1] for line in result[0]) / len(result[0]),
            'language': self.language,
            'positions': [line[0] for line in result[0]]
        }
```

### 导出器接口

#### 基础接口
```python
class Exporter(ABC):
    """导出器基础接口"""
    
    @abstractmethod
    def export(self, subtitles: List[Subtitle], output_path: str, **kwargs):
        """
        导出字幕文件
        
        Args:
            subtitles (List[Subtitle]): 字幕列表
            output_path (str): 输出路径
            **kwargs: 导出参数
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的格式列表
        
        Returns:
            List[str]: 格式列表
        """
        pass
```

#### 实现示例
```python
class SRTExporter(Exporter):
    """SRT格式导出器"""
    
    def export(self, subtitles: List[Subtitle], output_path: str, **kwargs):
        with open(output_path, 'w', encoding='utf-8') as f:
            for subtitle in subtitles:
                f.write(f"{subtitle.index}\n")
                f.write(f"{self._format_time(subtitle.start_time)} --> {self._format_time(subtitle.end_time)}\n")
                f.write(f"{subtitle.text}\n\n")
    
    def get_supported_formats(self) -> List[str]:
        return ['srt']
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
```

## 事件系统

### 信号系统

#### 主要信号
```python
# MainViewModel 信号
frame_changed = pyqtSignal(np.ndarray)           # 帧变化
config_changed = pyqtSignal(OcrConfig)           # 配置变化
error_occurred = pyqtSignal(str)                 # 错误发生
video_loaded = pyqtSignal(int)                   # 视频加载完成
subtitles_processed = pyqtSignal(list)          # 字幕处理完成

# EnhancedVideoPlayer 信号
frame_updated = pyqtSignal(np.ndarray)           # 帧更新
roi_selected = pyqtSignal(QRect)                # ROI选择
playback_state_changed = pyqtSignal(bool)       # 播放状态变化

# EnhancedOCRPreview 信号
result_selected = pyqtSignal(OCRResult)          # 结果选择
result_edited = pyqtSignal(OCRResult)            # 结果编辑
text_exported = pyqtSignal(str)                  # 文本导出
```

#### 信号连接示例
```python
# 连接信号
def setup_connections(self):
    """设置信号连接"""
    # 视图模型到UI
    self.vm.frame_changed.connect(self.video_player.update_frame)
    self.vm.error_occurred.connect(self.show_error_message)
    self.vm.subtitles_processed.connect(self.on_subtitles_ready)
    
    # UI到视图模型
    self.video_player.roi_selected.connect(self.vm.update_roi)
    self.ocr_preview.text_exported.connect(self.export_subtitles)
    
    # 内部组件连接
    self.roi_panel.roi_config_changed.connect(self.on_roi_config_changed)
```

### 自定义事件

#### 事件定义
```python
class CustomEvent(QEvent):
    """自定义事件基类"""
    
    def __init__(self, event_type: QEvent.Type, data: Any = None):
        super().__init__(event_type)
        self.data = data

# 事件类型
PROCESSING_STARTED = QEvent.Type(QEvent.registerEventType())
PROCESSING_PROGRESS = QEvent.Type(QEvent.registerEventType())
PROCESSING_COMPLETED = QEvent.Type(QEvent.registerEventType())
```

#### 事件发送示例
```python
def send_processing_event(self, event_type: QEvent.Type, data: Any):
    """发送处理事件"""
    event = CustomEvent(event_type, data)
    QApplication.postEvent(self, event)

def customEvent(self, event: QEvent):
    """处理自定义事件"""
    if event.type() == PROCESSING_STARTED:
        self.on_processing_started(event.data)
    elif event.type() == PROCESSING_PROGRESS:
        self.on_processing_progress(event.data)
    elif event.type() == PROCESSING_COMPLETED:
        self.on_processing_completed(event.data)
```

## 错误处理

### 错误类型

#### VisionSubError
```python
class VisionSubError(Exception):
    """VisionSub基础错误类"""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.datetime.now()

class VideoError(VisionSubError):
    """视频处理错误"""
    pass

class OCRError(VisionSubError):
    """OCR处理错误"""
    pass

class ConfigError(VisionSubError):
    """配置错误"""
    pass
```

### 错误处理策略

#### 全局错误处理
```python
class ErrorHandler:
    """全局错误处理器"""
    
    def __init__(self):
        self.error_log = []
        self.max_log_size = 1000
    
    def handle_error(self, error: Exception, context: str = None):
        """
        处理错误
        
        Args:
            error (Exception): 错误对象
            context (str): 错误上下文
        """
        error_info = {
            'timestamp': datetime.datetime.now(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_info)
        
        # 保持日志大小限制
        if len(self.error_log) > self.max_log_size:
            self.error_log = self.error_log[-self.max_log_size:]
        
        # 记录到文件
        self._log_to_file(error_info)
        
        # 发送错误信号
        self.error_occurred.emit(str(error))
    
    def _log_to_file(self, error_info: dict):
        """记录错误到文件"""
        try:
            log_file = Path.home() / ".visionsub" / "error.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{error_info['timestamp']}] {error_info['type']}: {error_info['message']}\n")
                if error_info['context']:
                    f.write(f"Context: {error_info['context']}\n")
                f.write(f"Traceback: {error_info['traceback']}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"Failed to log error: {e}")
```

## 最佳实践

### 1. 异步处理

#### 正确的异步处理
```python
async def process_video_async(self, video_path: str):
    """异步处理视频"""
    try:
        # 创建任务
        tasks = []
        
        # 分批处理
        for batch in self.create_batches(video_path):
            task = asyncio.create_task(self.process_batch(batch))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        subtitles = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            else:
                subtitles.extend(result)
        
        return subtitles
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise
```

### 2. 内存管理

#### 有效的内存管理
```python
class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        self.cache = LRUCache(maxsize=100)
    
    def allocate_memory(self, size_mb: int) -> bool:
        """
        分配内存
        
        Args:
            size_mb (int): 需要的内存大小（MB）
            
        Returns:
            bool: 是否分配成功
        """
        if self.current_memory_mb + size_mb <= self.max_memory_mb:
            self.current_memory_mb += size_mb
            return True
        else:
            # 尝试清理缓存
            self._cleanup_cache()
            return self.current_memory_mb + size_mb <= self.max_memory_mb
    
    def release_memory(self, size_mb: int):
        """释放内存"""
        self.current_memory_mb = max(0, self.current_memory_mb - size_mb)
    
    def _cleanup_cache(self):
        """清理缓存"""
        # 清理50%的缓存
        cleanup_size = len(self.cache) // 2
        for _ in range(cleanup_size):
            if self.cache:
                self.cache.popitem(last=False)
```

### 3. 线程安全

#### 线程安全的实现
```python
class ThreadSafeROIManager:
    """线程安全的ROI管理器"""
    
    def __init__(self):
        self.rois = {}
        self.lock = threading.RLock()
    
    def add_roi(self, roi_id: str, roi: ROIInfo):
        """线程安全地添加ROI"""
        with self.lock:
            self.rois[roi_id] = roi
    
    def get_roi(self, roi_id: str) -> Optional[ROIInfo]:
        """线程安全地获取ROI"""
        with self.lock:
            return self.rois.get(roi_id)
    
    def remove_roi(self, roi_id: str):
        """线程安全地删除ROI"""
        with self.lock:
            if roi_id in self.rois:
                del self.rois[roi_id]
```

### 4. 配置管理

#### 配置管理最佳实践
```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = AppConfig()
        self.config_file = Path.home() / ".visionsub" / "config.json"
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        try:
            if self.config_file.exists():
                self.config = AppConfig.load_from_file(str(self.config_file))
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            self.config = AppConfig()
    
    def save_config(self):
        """保存配置"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.save_to_file(str(self.config_file))
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
```

### 5. 日志记录

#### 结构化日志记录
```python
class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 添加文件处理器
        file_handler = logging.FileHandler(
            Path.home() / ".visionsub" / "logs" / f"{name}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_event(self, event_type: str, data: dict):
        """记录结构化事件"""
        log_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
```

---

*本文档涵盖了VisionSub的主要API接口和使用方法。如有任何疑问，请参考代码示例或联系开发团队。*
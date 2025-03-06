# 视频OCR字幕提取工具

基于Tesseract OCR的视频字幕提取工具，能够自动识别视频中的字幕并生成SRT格式的字幕文件。

## 功能特点

- 支持多种视频格式（MP4, AVI, MKV等）
- 基于Tesseract OCR引擎进行文字识别
- 支持中文、英文、日文等多种语言
- 可自定义字幕区域（上方、下方、全屏等）
- 自动过滤相似帧，提高处理效率
- 可调整字幕提取间隔
- 友好的图形用户界面，支持拖放操作
- 可作为命令行工具使用

## 系统要求

- Python 3.8+
- Tesseract OCR引擎
- 依赖库：
  - OpenCV
  - NumPy
  - Pillow
  - pytesseract
  - pysrt
  - tqdm
  - tkinterdnd2（可选，用于拖放功能）
  - scikit-image（可选，用于图像相似度计算）

## 安装说明

### 1. 安装Tesseract OCR

#### macOS
```bash
brew install tesseract
brew install tesseract-lang  # 安装额外语言包
```

#### Ubuntu/Debian
```bash
sudo apt install tesseract-ocr libtesseract-dev
```

#### Windows
从 [Tesseract下载页面](https://github.com/UB-Mannheim/tesseract/wiki) 下载并安装。

### 2. 安装依赖库

```bash
# 使用安装脚本一键安装所有依赖
bash install_deps.sh

# 或者手动安装
pip install -r requirements.txt
```

## 使用方法

### 图形界面

启动图形界面：

```bash
bash run_app.sh
```

然后可以：
1. 拖放视频文件或点击选择文件
2. 设置OCR语言和字幕区域
3. 设置提取间隔
4. 选择输出目录
5. 点击"开始提取"按钮

### 命令行

通过命令行直接提取视频字幕：

```bash
python main.py 视频文件路径 -o 输出文件路径 --lang chi_sim --area 0.7,1.0 --interval 1.0
```

参数说明：
- `--lang`：语言代码，如`chi_sim`（简体中文）、`eng`（英文）
- `--area`：字幕区域，格式为"开始高度,结束高度"，范围0.0-1.0
- `--interval`：提取帧的时间间隔（秒）

## 故障排除

### 提示"No module named 'xxx'"

```bash
bash install_deps.sh
```

### Tesseract OCR无法使用

确保Tesseract已正确安装并添加到系统PATH中。运行以下命令验证安装：

```bash
tesseract --version
```

### 图像识别效果不理想

- 尝试调整字幕区域，缩小范围
- 减小提取间隔，增加采样密度
- 检查视频字幕清晰度

## 开发者信息

本工具基于Python和Tesseract OCR开发，使用tkinter构建图形界面。核心模块包括：

- `main.py`：主程序及命令行接口
- `ocr_utils.py`：OCR引擎封装
- `video_utils.py`：视频处理工具
- `video_upload_interface.py`：图形用户界面

## 许可证

MIT License
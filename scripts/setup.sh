#!/bin/bash

# 视频OCR字幕提取工具 - Mac安装脚本

echo "=== 视频OCR字幕提取工具安装脚本 ==="
echo "该脚本将帮助您在Mac系统上安装所需的依赖"

# 检查是否安装了Homebrew
if ! command -v brew &> /dev/null; then
    echo "正在安装Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew已安装，正在更新..."
    brew update
fi

# 安装Python
if ! command -v python3 &> /dev/null; then
    echo "正在安装Python..."
    brew install python
else
    echo "Python已安装"
fi

# 安装FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "正在安装FFmpeg..."
    brew install ffmpeg
else
    echo "FFmpeg已安装"
fi

# 安装OpenCV依赖
echo "正在安装OpenCV依赖..."
brew install cmake pkg-config

# 创建并激活Python虚拟环境
echo "正在创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装Python依赖
echo "正在安装Python依赖..."
pip install --upgrade pip
pip install -e .

echo "=== 安装完成！ ==="
echo "您可以通过以下命令激活虚拟环境并运行应用："
echo "source venv/bin/activate"
echo "python -m visionsub.app" 
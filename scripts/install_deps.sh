#!/bin/bash

# 视频OCR字幕提取工具 - 依赖安装脚本

echo "=== 视频OCR字幕提取工具依赖安装 ==="
echo "该脚本将安装所有必要的依赖项"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.8+"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip，请先安装pip"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
echo "Python版本: $PYTHON_VERSION"

# 获取系统架构和操作系统信息
ARCH=$(uname -m)
OS_TYPE=$(uname)
echo "系统架构: $ARCH, 操作系统: $OS_TYPE"

# 处理PEP 668兼容性
# 新版Python可能需要--break-system-packages参数
PIP_EXTRA_ARGS=""
if python3 -m pip install --help | grep -q -- "--break-system-packages"; then
    echo "检测到PEP 668支持，添加--break-system-packages参数"
    PIP_EXTRA_ARGS="--break-system-packages"
fi

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip $PIP_EXTRA_ARGS

# 安装setuptools (打包必需)
echo "安装setuptools..."
python3 -m pip install --upgrade setuptools wheel $PIP_EXTRA_ARGS

# 安装pillow (图像处理库)
echo "安装Pillow..."
python3 -m pip install --upgrade pillow $PIP_EXTRA_ARGS

# 安装opencv (视频处理)
echo "安装OpenCV..."
python3 -m pip install --upgrade opencv-python $PIP_EXTRA_ARGS

# 安装numpy (数值计算)
echo "安装NumPy..."
python3 -m pip install --upgrade numpy $PIP_EXTRA_ARGS

# 安装pysrt (SRT字幕处理)
echo "安装pysrt..."
python3 -m pip install --upgrade pysrt $PIP_EXTRA_ARGS

# 安装tkinterdnd2 (拖放功能)
echo "安装tkinterdnd2..."
python3 -m pip install --upgrade tkinterdnd2 $PIP_EXTRA_ARGS

# 安装tqdm (进度条库)
echo "安装tqdm..."
python3 -m pip install --upgrade tqdm $PIP_EXTRA_ARGS

# 安装scikit-image (可选依赖 - 图像相似度计算)
echo "安装scikit-image (可选)..."
python3 -m pip install --upgrade scikit-image $PIP_EXTRA_ARGS || {
    echo "警告: scikit-image 安装失败，程序将使用替代方法进行图像相似度计算"
}

# 检查Tesseract是否已安装
check_tesseract() {
    if command -v tesseract &> /dev/null; then
        echo "Tesseract已安装，版本:"
        tesseract --version | head -n 1
        return 0
    fi
    
    # 提示用户安装Tesseract
    echo "未检测到Tesseract OCR。这是一个需要系统级安装的OCR引擎。"
    
    if [ "$OS_TYPE" = "Darwin" ]; then
        echo "在macOS上，您可以使用Homebrew安装Tesseract:"
        echo "  brew install tesseract"
        echo "  brew install tesseract-lang  # 安装额外语言包"
        
        # 检查是否安装了Homebrew
        if command -v brew &> /dev/null; then
            echo "检测到Homebrew，是否现在安装Tesseract? (y/n)"
            read -n 1 -r REPLY
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "安装Tesseract..."
                brew install tesseract
                brew install tesseract-lang
                echo "Tesseract安装完成！"
                return 0
            fi
        else
            echo "未检测到Homebrew，请先安装Homebrew: https://brew.sh/"
        fi
    elif [ "$OS_TYPE" = "Linux" ]; then
        echo "在Linux上，您可以使用包管理器安装Tesseract:"
        echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr libtesseract-dev"
        echo "  Fedora: sudo dnf install tesseract tesseract-devel"
        echo "  Arch Linux: sudo pacman -S tesseract tesseract-data-eng"
        
        # 检测Linux发行版
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" = "ubuntu" || "$ID" = "debian" ]]; then
                echo "检测到Ubuntu/Debian，是否现在安装Tesseract? (y/n)"
                read -n 1 -r REPLY
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "安装Tesseract..."
                    sudo apt-get update
                    sudo apt-get install -y tesseract-ocr libtesseract-dev
                    echo "Tesseract安装完成！"
                    return 0
                fi
            elif [ "$ID" = "fedora" ]; then
                echo "检测到Fedora，是否现在安装Tesseract? (y/n)"
                read -n 1 -r REPLY
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "安装Tesseract..."
                    sudo dnf install -y tesseract tesseract-devel
                    echo "Tesseract安装完成！"
                    return 0
                fi
            elif [ "$ID" = "arch" ]; then
                echo "检测到Arch Linux，是否现在安装Tesseract? (y/n)"
                read -n 1 -r REPLY
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "安装Tesseract..."
                    sudo pacman -S --noconfirm tesseract tesseract-data-eng
                    echo "Tesseract安装完成！"
                    return 0
                fi
            fi
        fi
    elif [[ "$OS_TYPE" == *"MINGW"* || "$OS_TYPE" == *"MSYS"* ]]; then
        echo "在Windows上，请访问以下网址下载安装Tesseract:"
        echo "  https://github.com/UB-Mannheim/tesseract/wiki"
        echo "安装后请确保将Tesseract添加到系统PATH中"
    fi
    
    echo "请手动安装Tesseract OCR，然后再继续。"
    return 1
}

# 安装pytesseract (Python的Tesseract接口)
install_pytesseract() {
    echo "安装pytesseract (Python的Tesseract接口)..."
    python3 -m pip install pytesseract $PIP_EXTRA_ARGS
    
    # 验证安装
    if python3 -c "import pytesseract" &> /dev/null; then
        echo "pytesseract安装成功 ✓"
        return 0
    else
        echo "pytesseract安装失败 ✗"
        return 1
    fi
}

# 安装Tesseract OCR
echo "=== 设置OCR引擎 ==="
echo "设置Tesseract OCR引擎..."
check_tesseract
install_pytesseract

# 安装主要依赖
echo "安装其他依赖..."
if [ -f "pyproject.toml" ]; then
    # 安装项目依赖
    echo "处理项目依赖..."
    python3 -m pip install -e . $PIP_EXTRA_ARGS || echo "部分依赖安装失败，程序可能无法正常运行"
else
    echo "警告: 未找到pyproject.toml文件"
fi

# 验证安装
echo "验证依赖安装..."
python3 -c "
import sys
try:
    import PIL
    print('PIL/Pillow ✓')
    import cv2
    print('OpenCV ✓')
    import numpy
    print('NumPy ✓')
    import pysrt
    print('pysrt ✓')
    import tqdm
    print('tqdm ✓')
    from tkinter import *
    print('Tkinter ✓')
    
    # 验证Tesseract
    try:
        import pytesseract
        print('Pytesseract ✓')
        # 尝试验证tesseract是否可用
        try:
            version = pytesseract.get_tesseract_version()
            print(f'  Tesseract版本: {version}')
        except Exception as e:
            print(f'  警告: Tesseract可能未正确安装: {e}')
            print('  请确保系统中安装了Tesseract OCR引擎')
    except ImportError:
        print('警告: Pytesseract 未安装 ✗')
    
    # 验证可选依赖
    try:
        import skimage
        from skimage.metrics import structural_similarity
        print('scikit-image ✓ (可选)')
    except ImportError:
        print('scikit-image ✗ (可选，将使用替代方法)')
    
    try:
        from tkinterdnd2 import *
        print('tkinterdnd2 ✓')
    except ImportError:
        print('tkinterdnd2 ✗ (可选，拖放功能不可用)')
except ImportError as e:
    print(f'错误: {e}')
    print('一些依赖未能成功安装')
    sys.exit(1)
"

# 如果验证失败
if [ $? -ne 0 ]; then
    echo "警告: 依赖验证失败。某些功能可能无法正常工作。"
else
    echo "=== 基本依赖安装完成 ==="
fi

echo "
=== OCR引擎配置 ===
OCR引擎: Tesseract OCR

您需要确保Tesseract OCR引擎在系统中正确安装:
- macOS: brew install tesseract tesseract-lang
- Ubuntu/Debian: sudo apt-get install tesseract-ocr
- Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装

=== 安装完成 ===
您现在可以运行应用：bash run_app.sh
"

# 是否需要创建Python虚拟环境
if [ ! -d "venv" ]; then
    echo "您是否想要创建Python虚拟环境？这有助于隔离依赖环境 (y/n)"
    read -n 1 -r REPLY
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "创建Python虚拟环境..."
        python3 -m venv venv
        echo "激活环境并重新运行安装脚本以安装依赖到虚拟环境中。"
        echo "在Linux/Mac上运行: source venv/bin/activate && bash install_deps.sh"
        echo "在Windows上运行: venv\\Scripts\\activate && install_deps.sh"
    fi
fi
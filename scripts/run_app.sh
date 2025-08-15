#!/bin/bash

# 视频OCR字幕提取工具 - 启动脚本

echo "=== 视频OCR字幕提取工具 ==="

# 检查Python版本
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)
        
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
            echo "警告: 推荐使用Python 3.8或更高版本，当前版本: $PYTHON_VERSION"
        else
            echo "Python版本: $PYTHON_VERSION"
        fi
    else
        echo "错误: 未找到Python 3"
        exit 1
    fi
}

# 获取系统架构
get_system_info() {
    OS_TYPE=$(uname)
    ARCH=$(uname -m)
    echo "操作系统: $OS_TYPE, 架构: $ARCH"
}

# 检查Tesseract是否已安装
check_tesseract() {
    if command -v tesseract &> /dev/null; then
        echo "Tesseract OCR已安装"
        tesseract --version | head -n 1
        return 0
    else
        echo "错误: 未检测到Tesseract OCR"
        echo "请先安装Tesseract OCR引擎:"
        
        if [ "$(uname)" == "Darwin" ]; then
            echo "  macOS: brew install tesseract tesseract-lang"
        elif [ "$(uname)" == "Linux" ]; then
            echo "  Ubuntu/Debian: sudo apt install tesseract-ocr"
            echo "  Fedora: sudo dnf install tesseract"
            echo "  Arch Linux: sudo pacman -S tesseract"
        else
            echo "  Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装"
        fi
        return 1
    fi
}

# 确保依赖安装
install_dependencies() {
    echo "检查并安装必要依赖..."
    
    # 使用--break-system-packages解决PEP 668问题
    PIP_EXTRA_ARGS=""
    if python3 -m pip install --help | grep -q -- "--break-system-packages"; then
        PIP_EXTRA_ARGS="--break-system-packages"
    fi
    
    # 升级pip
    echo "升级pip..."
    python3 -m pip install --upgrade pip $PIP_EXTRA_ARGS
    
    # 安装关键依赖
    echo "安装核心依赖..."
    python3 -m pip install --upgrade pillow opencv-python tqdm numpy pysrt $PIP_EXTRA_ARGS
    
    # 尝试安装tkinterdnd2
    echo "安装tkinterdnd2(拖放支持)..."
    python3 -m pip install --upgrade tkinterdnd2 $PIP_EXTRA_ARGS
    
    # 安装pytesseract
    echo "安装pytesseract(Tesseract OCR接口)..."
    python3 -m pip install --upgrade pytesseract $PIP_EXTRA_ARGS
    
    # 尝试安装scikit-image (可选)
    echo "安装scikit-image(可选依赖)..."
    python3 -m pip install --upgrade scikit-image $PIP_EXTRA_ARGS || echo "scikit-image安装失败，将使用替代方法"
    
    # 安装项目依赖
    if [ -f "pyproject.toml" ]; then
        echo "安装项目依赖..."
        python3 -m pip install -e . $PIP_EXTRA_ARGS
    else
        echo "警告: 未找到pyproject.toml文件"
    fi
    
    echo "依赖安装完成"
}

# 检查依赖
check_dependencies() {
    MISSING_DEPS=()
    
    # 检查PIL
    if ! python3 -c "import PIL" &> /dev/null; then
        MISSING_DEPS+=("pillow")
    fi
    
    # 检查OpenCV
    if ! python3 -c "import cv2" &> /dev/null; then
        MISSING_DEPS+=("opencv-python")
    fi
    
    # 检查tqdm
    if ! python3 -c "import tqdm" &> /dev/null; then
        MISSING_DEPS+=("tqdm")
    fi
    
    # 检查numpy
    if ! python3 -c "import numpy" &> /dev/null; then
        MISSING_DEPS+=("numpy")
    fi
    
    # 检查pytesseract
    if ! python3 -c "import pytesseract" &> /dev/null; then
        MISSING_DEPS+=("pytesseract")
    fi
    
    # 检查pysrt
    if ! python3 -c "import pysrt" &> /dev/null; then
        MISSING_DEPS+=("pysrt")
    fi
    
    # 检查scikit-image (可选)
    if ! python3 -c "import skimage" &> /dev/null; then
        echo "注意: 未检测到scikit-image，将使用替代方法处理图像相似度计算"
        # 不将其添加到必需依赖列表，因为它是可选的
    fi
    
    # 检查tkinterdnd2 (可选)
    if ! python3 -c "import tkinterdnd2" &> /dev/null; then
        echo "注意: 未检测到tkinterdnd2，拖放功能将不可用"
        # 可选依赖，不添加到必需列表
    fi
    
    # 如果有缺失的依赖，安装它们
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "检测到缺少的依赖: ${MISSING_DEPS[*]}"
        echo "是否自动安装依赖? (y/n)"
        read -n 1 -r REPLY
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_dependencies
        else
            echo "请手动安装缺少的依赖后再运行程序"
            echo "运行: pip install ${MISSING_DEPS[*]}"
            exit 1
        fi
    else
        echo "所有必需的Python依赖已安装"
    fi
}

# 尝试在没有Tesseract的情况下运行
run_without_tesseract() {
    echo "警告: 未检测到Tesseract OCR，但仍将尝试启动程序"
    echo "OCR识别功能可能无法正常工作"
    echo "是否继续? (y/n)"
    read -n 1 -r REPLY
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "继续启动..."
        export IGNORE_TESSERACT_ERROR=1
    else
        echo "请先安装所需依赖"
        echo "运行: bash install_deps.sh"
        exit 1
    fi
}

# 检查环境
check_python_version
get_system_info
check_dependencies

# 检查Tesseract安装
if ! check_tesseract; then
    run_without_tesseract
fi

# 检查Python环境
if [ -d "venv" ]; then
    # 激活虚拟环境
    echo "激活Python虚拟环境..."
    source venv/bin/activate
else
    echo "未找到虚拟环境，使用系统Python..."
fi

# 验证所有基本依赖都已安装
echo "验证依赖状态..."
python3 -c "
import sys
try:
    import PIL
    import cv2
    import tqdm
    import numpy
    import pysrt
    from tqdm import tqdm
    print('✓ 基本依赖已正确安装')
    
    # 尝试导入pytesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f'✓ Tesseract OCR已安装，版本: {version}')
    except ImportError as e:
        print(f'⚠️ pytesseract未安装: {e}')
        print('OCR功能可能无法正常使用')
    except Exception as e:
        print(f'⚠️ Tesseract OCR可能未正确安装: {e}')
        print('请确保系统中安装了Tesseract OCR引擎')
    
    # 检查可选依赖
    try:
        import skimage
        from skimage.metrics import structural_similarity
        print('✓ scikit-image已安装 (可选依赖)')
    except ImportError:
        print('ℹ️ scikit-image未安装 (可选依赖)，将使用替代方法')
        
    try:
        import tkinterdnd2
        print('✓ tkinterdnd2已安装 (拖放支持)')
    except ImportError:
        print('ℹ️ tkinterdnd2未安装 (可选依赖)，拖放功能不可用')
except ImportError as e:
    print(f'❌ 依赖验证失败: {e}')
    print('请运行: pip install pillow opencv-python tqdm numpy pytesseract pysrt')
    sys.exit(1)
"

# 启动应用
echo "启动视频OCR字幕提取工具..."
python3 -m visionsub.app

# 如果有错误，等待用户按键
if [ $? -ne 0 ]; then
    echo "应用运行失败，错误代码: $?"
    echo "按任意键退出..."
    read -n 1
fi 
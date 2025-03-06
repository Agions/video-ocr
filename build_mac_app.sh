#!/bin/bash

# 视频OCR字幕提取工具 - Mac应用打包综合脚本

echo "=== 视频OCR字幕提取工具 Mac应用打包脚本 ==="
echo "该脚本将引导您完成从Python源代码到DMG安装包的打包过程"

# 检查必要的工具
echo "检查必要的工具..."

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

# 检查Homebrew
if ! command -v brew &> /dev/null; then
    echo "警告: 未找到Homebrew，将尝试安装"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew已安装"
fi

# 先安装setuptools (修复打包错误)
echo "安装setuptools和wheel..."
python3 -m pip install --upgrade pip setuptools wheel

# 安装依赖
echo "安装必要的依赖..."
python3 -m pip install --upgrade pillow
python3 -m pip install -r requirements.txt

# 安装打包工具
echo "安装打包工具..."
python3 -m pip install --upgrade py2app
brew install create-dmg imagemagick

# 创建图标
echo "创建应用图标..."
chmod +x create_icon.sh
./create_icon.sh

# 创建setup.py文件
echo "创建setup.py文件..."
cat > setup.py << EOF
from setuptools import setup

APP = ['video_upload_interface.py']
DATA_FILES = [
    'main.py',
    'video_utils.py',
    'ocr_utils.py',
    'subtitle_utils.py',
    'extract_subtitle.py',
    'example.py',
    'README.md'
]
OPTIONS = {
    'argv_emulation': True,
    'packages': ['paddleocr', 'paddlepaddle', 'cv2', 'numpy', 'pysrt', 'skimage', 'tqdm', 'tkinter', 'PIL'],
    'includes': ['tkinter', 'PIL._tkinter_finder'],
    'iconfile': 'app_icon.icns',
    'plist': {
        'CFBundleName': '视频OCR字幕提取工具',
        'CFBundleDisplayName': '视频OCR字幕提取工具',
        'CFBundleGetInfoString': "视频OCR字幕提取工具",
        'CFBundleIdentifier': "com.videoocr.subtitleextractor",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0.0",
        'NSHumanReadableCopyright': "Copyright © 2023, 视频OCR字幕提取工具",
    },
    'resources': [],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
EOF

# 清理以前的构建
echo "清理以前的构建..."
rm -rf build dist

# 使用py2app构建应用
echo "构建Mac应用..."
python3 setup.py py2app

# 检查构建是否成功
if [ ! -d "dist/视频OCR字幕提取工具.app" ]; then
    echo "错误: 应用构建失败！"
    exit 1
fi

# 准备DMG内容
echo "准备DMG内容..."
mkdir -p dmg_content
cp -r dist/视频OCR字幕提取工具.app dmg_content/
cp README.md dmg_content/
ln -sf /Applications dmg_content/

# 创建DMG
echo "创建DMG文件..."
create-dmg \
  --volname "视频OCR字幕提取工具" \
  --volicon "app_icon.icns" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "视频OCR字幕提取工具.app" 150 190 \
  --hide-extension "视频OCR字幕提取工具.app" \
  --app-drop-link 450 190 \
  "视频OCR字幕提取工具.dmg" \
  dmg_content/

# 检查DMG是否创建成功
if [ ! -f "视频OCR字幕提取工具.dmg" ]; then
    echo "错误: DMG创建失败！"
    exit 1
fi

echo "=== 打包完成！ ==="
echo "DMG文件已创建: 视频OCR字幕提取工具.dmg"
echo "您可以将此文件分发给用户。"
echo "用户只需双击DMG文件，将应用拖到Applications文件夹即可完成安装。" 
#!/bin/bash

# 视频OCR字幕提取工具 - DMG打包脚本

echo "=== 视频OCR字幕提取工具 DMG打包脚本 ==="

# 安装必要的工具
echo "正在安装打包所需工具..."
pip install py2app
brew install create-dmg

# 创建setup.py文件
echo "正在创建setup.py文件..."
cat > setup.py << EOF
from setuptools import setup

APP = ['extract_subtitle.py']
DATA_FILES = [
    'main.py',
    'video_utils.py',
    'ocr_utils.py',
    'subtitle_utils.py',
    'example.py',
    'README.md'
]
OPTIONS = {
    'argv_emulation': True,
    'packages': ['paddleocr', 'paddlepaddle', 'cv2', 'numpy', 'pysrt', 'skimage', 'tqdm'],
    'iconfile': 'app_icon.icns',
    'plist': {
        'CFBundleName': '视频OCR字幕提取工具',
        'CFBundleDisplayName': '视频OCR字幕提取工具',
        'CFBundleGetInfoString': "视频OCR字幕提取工具",
        'CFBundleIdentifier': "com.videoocr.subtitleextractor",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0.0",
        'NSHumanReadableCopyright': "Copyright © 2023, 视频OCR字幕提取工具",
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
EOF

# 创建一个简单的图标 (如果没有的话)
echo "正在创建应用图标..."
mkdir -p app_icon.iconset
# 这里应该放置实际的图标生成代码，但因为我们没有实际的图标文件，所以跳过
# 如果有图标文件，可以使用iconutil命令将iconset转换为icns

# 清理以前的构建
echo "正在清理以前的构建..."
rm -rf build dist

# 使用py2app构建应用
echo "正在构建Mac应用..."
python setup.py py2app

# 准备DMG内容
echo "正在准备DMG内容..."
mkdir -p dmg_content
cp -r dist/视频OCR字幕提取工具.app dmg_content/
cp README.md dmg_content/
ln -sf /Applications dmg_content/

# 创建DMG
echo "正在创建DMG文件..."
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

echo "=== DMG打包完成！==="
echo "您可以在当前目录找到 视频OCR字幕提取工具.dmg 文件" 
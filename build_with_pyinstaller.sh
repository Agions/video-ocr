#!/bin/bash

# 视频OCR字幕提取工具 - 使用PyInstaller打包脚本

echo "=== 视频OCR字幕提取工具 PyInstaller打包脚本 ==="
echo "该脚本使用PyInstaller打包应用，并创建DMG安装包"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.8+"
    exit 1
fi

# 升级pip和setuptools
echo "升级pip和setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# 安装依赖
echo "安装必要的依赖..."
python3 -m pip install --upgrade pillow
python3 -m pip install --upgrade tqdm
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade pyinstaller

# 安装DMG创建工具
if ! command -v create-dmg &> /dev/null; then
    echo "安装create-dmg工具..."
    brew install create-dmg
fi

# 创建图标
echo "创建应用图标..."
chmod +x create_icon.sh
./create_icon.sh

# 清理以前的构建
echo "清理以前的构建..."
rm -rf build dist

# 使用PyInstaller打包
echo "使用PyInstaller打包应用..."
pyinstaller --windowed \
    --name "视频OCR字幕提取工具" \
    --icon=app_icon.icns \
    --add-data "README.md:." \
    --hidden-import=paddleocr \
    --hidden-import=paddlepaddle \
    --hidden-import=pysrt \
    --hidden-import=skimage \
    --hidden-import=PIL \
    --hidden-import=PIL._tkinter_finder \
    --hidden-import=tkinter \
    --collect-submodules=PIL \
    video_upload_interface.py

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

echo "=== 打包完成！ ==="
echo "DMG文件已创建: 视频OCR字幕提取工具.dmg" 
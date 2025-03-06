#!/bin/bash

# 创建基本的应用图标
# 注意：此脚本需要安装 imagemagick

echo "=== 创建应用图标 ==="

# 检查是否安装了ImageMagick
if ! command -v convert &> /dev/null; then
    echo "需要安装ImageMagick，正在安装..."
    brew install imagemagick
fi

# 创建图标目录
mkdir -p app_icon.iconset

# 创建一个简单的文本图标
echo "正在创建基本图标..."

# 不同尺寸的图标
sizes=(16 32 64 128 256 512 1024)

for size in "${sizes[@]}"; do
    # 创建图标
    convert -size "${size}x${size}" xc:none \
        -fill '#3498db' -draw "roundrectangle 0,0,${size},${size},${size}/5,${size}/5" \
        -fill white \
        -pointsize $((size/3)) -gravity center -draw "text 0,0 'OCR'" \
        "app_icon.iconset/icon_${size}x${size}.png"
    
    # 创建2x版本
    double=$((size*2))
    if [[ "${size}" -le 512 ]]; then
        convert -size "${double}x${double}" xc:none \
            -fill '#3498db' -draw "roundrectangle 0,0,${double},${double},${double}/5,${double}/5" \
            -fill white \
            -pointsize $((double/3)) -gravity center -draw "text 0,0 'OCR'" \
            "app_icon.iconset/icon_${size}x${size}@2x.png"
    fi
done

# 使用iconutil将iconset转换为icns
iconutil -c icns app_icon.iconset

echo "=== 图标创建完成 ==="
echo "图标文件已保存为 app_icon.icns" 
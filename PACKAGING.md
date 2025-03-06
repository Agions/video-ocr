# 视频OCR字幕提取工具打包指南

本文档提供了将视频OCR字幕提取工具打包为Mac DMG安装包的详细步骤。

## 准备工作

在开始之前，确保您的系统满足以下要求：

1. macOS 10.15 (Catalina) 或更高版本
2. Python 3.8+
3. Homebrew 包管理器
4. 足够的磁盘空间（至少需要2GB左右）

## 打包方法一：使用py2app

### 步骤1：安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/video-ocr-tool.git
cd video-ocr-tool

# 安装项目依赖
pip install -r requirements.txt

# 安装打包工具
pip install py2app
brew install create-dmg imagemagick
```

### 步骤2：创建应用图标

```bash
# 使用提供的脚本创建图标
chmod +x create_icon.sh
./create_icon.sh
```

### 步骤3：打包为DMG

```bash
# 使用提供的打包脚本
chmod +x build_mac_app.sh
./build_mac_app.sh
```

打包完成后，您将在当前目录找到 `视频OCR字幕提取工具.dmg` 文件。

## 打包方法二：使用PyInstaller（推荐）

PyInstaller通常比py2app提供更好的兼容性，推荐使用此方法。

```bash
# 使用PyInstaller打包脚本
chmod +x build_with_pyinstaller.sh
./build_with_pyinstaller.sh
```

## 安装说明

将打包好的DMG分发给用户后，用户安装步骤如下：

1. 双击 `视频OCR字幕提取工具.dmg` 文件打开
2. 将 `视频OCR字幕提取工具.app` 拖到 `Applications` 文件夹
3. 在应用程序列表中找到并运行应用

## 常见问题

### 1. 安全性警告

首次运行时，macOS可能会显示"无法验证开发者"的警告。解决方法：

1. 右键点击应用
2. 选择"打开"
3. 在弹出的对话框中点击"打开"

### 2. 打包过程中出现错误

如果在打包过程中遇到问题：

- 确保所有依赖都已正确安装
- 检查日志输出，查找具体错误
- 尝试使用另一种打包方法

### 3. 包大小问题

打包后的DMG文件可能比较大（约500MB-1GB），这是因为它包含了完整的Python环境和所有依赖库。如需优化大小，可以在打包脚本中添加以下选项：

```
--exclude-module matplotlib
--exclude-module scipy
```

## 支持与反馈

如有问题，请提交GitHub Issues或发送邮件至[您的邮箱]。 
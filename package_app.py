#!/usr/bin/env python3
"""
VisionSub Packaging Script
打包DMG和EXE客户端的脚本

作者: Agions
版本: 1.0.0
日期: 2025-08-15
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"   命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 错误: {e}")
        if e.stderr:
            print(f"   错误输出: {e.stderr}")
        return False

def check_dependencies():
    """检查打包依赖"""
    print("📦 检查打包依赖...")
    
    dependencies = [
        ("pyinstaller", "PyInstaller"),
        ("create-dmg", "create-dmg (macOS)"),
        ("iconutil", "iconutil (macOS)"),
    ]
    
    missing = []
    for dep, desc in dependencies:
        try:
            if dep == "create-dmg":
                # 检查create-dmg是否安装
                result = subprocess.run(["which", "create-dmg"], capture_output=True, text=True)
                if result.returncode != 0:
                    missing.append(desc)
            elif dep == "iconutil":
                # 检查iconutil是否可用 (macOS系统工具)
                result = subprocess.run(["which", "iconutil"], capture_output=True, text=True)
                if result.returncode != 0:
                    missing.append(desc)
            else:
                # 检查Python包 - 使用subprocess而不是import
                if dep == "pyinstaller":
                    result = subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], capture_output=True, text=True)
                else:
                    result = subprocess.run([sys.executable, "-c", f"import {dep}"], capture_output=True, text=True)
                if result.returncode != 0:
                    missing.append(desc)
        except subprocess.CalledProcessError:
            missing.append(desc)
    
    if missing:
        print(f"   ❌ 缺少依赖: {', '.join(missing)}")
        print("   💡 请安装缺少的依赖:")
        if "PyInstaller" in missing:
            print("      pip install pyinstaller")
        if "create-dmg" in missing:
            print("      brew install create-dmg")
        return False
    
    print("   ✅ 所有依赖已安装")
    return True

def package_macos_dmg():
    """打包macOS DMG"""
    print("🍎 开始打包macOS DMG...")
    
    # 创建构建目录
    build_dir = Path("build")
    dist_dir = Path("dist")
    assets_dir = Path("assets")
    
    # 清理之前的构建
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # 使用PyInstaller构建
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name=VisionSub",
        f"--icon={assets_dir}/app_icon.icns",
        "--add-data=assets:assets",
        "--add-data=src/visionsub/ui/themes:visionsub/ui/themes",
        "--hidden-import=PIL",
        "--hidden-import=paddleocr",
        "--hidden-import=pytesseract",
        "--hidden-import=easyocr",
        "run_gui.py"
    ]
    
    if not run_command(pyinstaller_cmd, "使用PyInstaller构建应用程序"):
        return False
    
    # 创建DMG
    app_path = dist_dir / "VisionSub.app"
    if not app_path.exists():
        print(f"   ❌ 应用程序未找到: {app_path}")
        return False
    
    # 创建DMG目录结构
    dmg_dir = build_dir / "dmg"
    dmg_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制应用程序到DMG目录
    shutil.copytree(app_path, dmg_dir / "VisionSub.app")
    
    # 创建符号链接
    (dmg_dir / "Applications").symlink_to("/Applications")
    
    # 创建DMG
    dmg_path = dist_dir / "VisionSub-2.0.0-macOS.dmg"
    if dmg_path.exists():
        dmg_path.unlink()
    
    # 使用hdiutil创建DMG
    create_dmg_cmd = [
        "hdiutil", "create",
        "-volname", "VisionSub 2.0.0",
        "-volname", "VisionSub",
        "-srcfolder", str(dmg_dir),
        "-ov", "-format", "UDZO",
        str(dmg_path)
    ]
    
    if not run_command(create_dmg_cmd, "创建DMG镜像"):
        return False
    
    print(f"   ✅ DMG创建成功: {dmg_path}")
    return True

def package_windows_exe():
    """打包Windows EXE"""
    print("🪟 开始打包Windows EXE...")
    
    # 创建构建目录
    build_dir = Path("build")
    dist_dir = Path("dist")
    assets_dir = Path("assets")
    
    # 清理之前的构建
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # 使用PyInstaller构建
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name=VisionSub",
        f"--icon={assets_dir}/windows/app_icon.ico",
        "--add-data=assets;assets",
        "--add-data=src/visionsub/ui/themes;visionsub/ui/themes",
        "--hidden-import=PIL",
        "--hidden-import=paddleocr",
        "--hidden-import=pytesseract",
        "--hidden-import=easyocr",
        "--uac-admin",  # 请求管理员权限
        "--version-file=version.txt",
        "run_gui.py"
    ]
    
    if not run_command(pyinstaller_cmd, "使用PyInstaller构建应用程序"):
        return False
    
    # 创建安装程序目录
    installer_dir = dist_dir / "installer"
    installer_dir.mkdir(exist_ok=True)
    
    # 复制EXE文件
    exe_path = dist_dir / "VisionSub.exe"
    if exe_path.exists():
        shutil.copy2(exe_path, installer_dir / "VisionSub.exe")
    else:
        print(f"   ❌ EXE文件未找到: {exe_path}")
        return False
    
    # 复制其他文件
    shutil.copytree(assets_dir, installer_dir / "assets", dirs_exist_ok=True)
    
    # 创建README文件
    readme_content = """# VisionSub 2.0.0

专业的视频OCR字幕提取工具

## 安装说明
1. 双击 VisionSub.exe 运行应用程序
2. 如果遇到安全警告，请点击"更多信息"然后选择"仍要运行"

## 系统要求
- Windows 10/11
- 4GB RAM (推荐8GB)
- 2GB 可用磁盘空间

## 功能特性
- 🎬 多格式视频支持
- 🔍 智能OCR识别
- 🎨 现代化界面
- ⚡ 高性能处理
- 🔒 安全可靠

## 技术支持
如有问题，请联系技术支持。
"""
    
    with open(installer_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"   ✅ EXE创建成功: {exe_path}")
    print(f"   📦 安装程序目录: {installer_dir}")
    return True

def create_version_file():
    """创建版本文件"""
    version_content = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(2, 0, 0, 0),
    prodvers=(2, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'VisionSub'),
           StringStruct(u'FileDescription', u'VisionSub - Professional Video OCR Tool'),
           StringStruct(u'FileVersion', u'2.0.0'),
           StringStruct(u'InternalName', u'VisionSub'),
           StringStruct(u'LegalCopyright', u'Copyright © 2025 VisionSub'),
           StringStruct(u'OriginalFilename', u'VisionSub.exe'),
           StringStruct(u'ProductName', u'VisionSub'),
           StringStruct(u'ProductVersion', u'2.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', 1033, 1200)])
  ]
)
"""
    
    with open("version.txt", "w", encoding="utf-8") as f:
        f.write(version_content)
    
    print("   ✅ 版本文件已创建")

def main():
    """主函数"""
    print("📦 VisionSub 打包工具")
    print("=" * 50)
    
    # 检查系统
    current_system = platform.system()
    print(f"   🖥️  当前系统: {current_system}")
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请安装缺少的依赖后重试")
        return 1
    
    # 创建版本文件
    create_version_file()
    
    success = False
    
    # 根据系统打包
    if current_system == "Darwin":  # macOS
        success = package_macos_dmg()
    elif current_system == "Windows":
        success = package_windows_exe()
    else:
        print(f"❌ 不支持的系统: {current_system}")
        return 1
    
    if success:
        print("\n✅ 打包完成!")
        print("📋 输出文件:")
        
        if current_system == "Darwin":
            dmg_path = Path("dist/VisionSub-2.0.0-macOS.dmg")
            if dmg_path.exists():
                print(f"   🍎 {dmg_path}")
        elif current_system == "Windows":
            exe_path = Path("dist/VisionSub.exe")
            installer_dir = Path("dist/installer")
            if exe_path.exists():
                print(f"   🪟 {exe_path}")
            if installer_dir.exists():
                print(f"   📦 {installer_dir}")
        
        return 0
    else:
        print("\n❌ 打包失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
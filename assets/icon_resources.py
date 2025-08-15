"""
VisionSub Icon Resources
自动生成的图标资源文件

作者: Agions
版本: 1.0.0
日期: 2025-08-15
"""

from pathlib import Path
from typing import Dict, List, Optional

class IconResources:
    """图标资源管理器"""
    
    def __init__(self, assets_dir: str = "assets"):
        self.assets_dir = Path(assets_dir)
        self.icon_cache = {}
    
    def get_app_icon_path(self, size: int = 64) -> str:
        """获取应用图标路径"""
        return str(self.assets_dir / "app_icons" / f"app_icon_{size}x{size}.png")
    
    def get_toolbar_icon_path(self, icon_type: str, size: int = 24) -> str:
        """获取工具栏图标路径"""
        return str(self.assets_dir / "toolbar" / icon_type / f"{icon_type}_{size}x{size}.png")
    
    def get_status_icon_path(self, status_type: str, size: int = 16) -> str:
        """获取状态图标路径"""
        return str(self.assets_dir / "status" / status_type / f"{status_type}_{size}x{size}.png")
    
    def get_macos_iconset_path(self) -> str:
        """获取macOS图标集路径"""
        return str(self.assets_dir / "macos" / "AppIcon.iconset")
    
    def get_windows_icon_path(self) -> str:
        """获取Windows图标路径"""
        return str(self.assets_dir / "windows" / "app_icon.ico")
    
    def get_web_favicon_path(self) -> str:
        """获取网站Favicon路径"""
        return str(self.assets_dir / "web" / "favicon.ico")
    
    def list_available_icons(self) -> Dict[str, List[str]]:
        """列出所有可用图标"""
        available_icons = {
            "app_icons": [],
            "toolbar": [],
            "status": []
        }
        
        # 应用图标
        app_icon_dir = self.assets_dir / "app_icons"
        if app_icon_dir.exists():
            available_icons["app_icons"] = [f.name for f in app_icon_dir.glob("*.png")]
        
        # 工具栏图标
        toolbar_dir = self.assets_dir / "toolbar"
        if toolbar_dir.exists():
            for icon_type in toolbar_dir.iterdir():
                if icon_type.is_dir():
                    available_icons["toolbar"].append(icon_type.name)
        
        # 状态图标
        status_dir = self.assets_dir / "status"
        if status_dir.exists():
            for status_type in status_dir.iterdir():
                if status_type.is_dir():
                    available_icons["status"].append(status_type.name)
        
        return available_icons

# 全局图标资源实例
icon_resources = IconResources()

# 便捷函数
def get_app_icon(size: int = 64) -> str:
    """获取应用图标路径"""
    return icon_resources.get_app_icon_path(size)

def get_toolbar_icon(icon_type: str, size: int = 24) -> str:
    """获取工具栏图标路径"""
    return icon_resources.get_toolbar_icon_path(icon_type, size)

def get_status_icon(status_type: str, size: int = 16) -> str:
    """获取状态图标路径"""
    return icon_resources.get_status_icon_path(status_type, size)

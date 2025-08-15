"""
Enhanced UI/UX system with responsive design and theme management
"""
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, pyqtSignal, QObject, 
    QTimer, QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QColor, QPalette, QFont, QFontDatabase, QIcon, 
    QPainter, QBrush, QPen, QLinearGradient, QRadialGradient
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QStyle, QStyleOption, QLayout,
    QSizePolicy, QSpacerItem, QFrame, QHBoxLayout, QVBoxLayout,
    QLayoutItem
)


class ThemeMode(Enum):
    """Theme modes"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"


class ColorRole(Enum):
    """Color roles for themes"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ACCENT = "accent"
    BACKGROUND = "background"
    SURFACE = "surface"
    TEXT_PRIMARY = "text_primary"
    TEXT_SECONDARY = "text_secondary"
    BORDER = "border"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


class SizeVariant(Enum):
    """Size variants for responsive design"""
    MOBILE = "mobile"      # < 768px
    TABLET = "tablet"        # 768px - 1024px
    DESKTOP = "desktop"      # 1024px - 1440px
    LARGE = "large"          # > 1440px


@dataclass
class ThemeColors:
    """Theme color definitions"""
    primary: QColor
    secondary: QColor
    accent: QColor
    background: QColor
    surface: QColor
    text_primary: QColor
    text_secondary: QColor
    border: QColor
    success: QColor
    warning: QColor
    error: QColor
    info: QColor
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization"""
        return {
            role.value: getattr(self, role.value).name()
            for role in ColorRole
        }


@dataclass
class ThemeTypography:
    """Typography definitions"""
    font_family: str = "Inter, system-ui, -apple-system, sans-serif"
    font_size_base: int = 14
    font_size_small: int = 12
    font_size_large: int = 16
    font_size_xlarge: int = 20
    line_height: float = 1.5
    letter_spacing: float = 0.0
    
    def get_font(self, size: int = None, weight: int = None) -> QFont:
        """Get font with specified size and weight"""
        font = QFont(self.font_family)
        
        if size:
            font.setPointSize(size)
        else:
            font.setPointSize(self.font_size_base)
            
        if weight:
            font.setWeight(weight)
            
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, self.letter_spacing)
        return font


@dataclass
class ThemeSpacing:
    """Spacing definitions"""
    unit: int = 8
    padding_small: int = 8
    padding_medium: int = 16
    padding_large: int = 24
    padding_xlarge: int = 32
    margin_small: int = 8
    margin_medium: int = 16
    margin_large: int = 24
    margin_xlarge: int = 32
    border_radius: int = 8
    border_radius_small: int = 4
    border_radius_large: int = 12


@dataclass
class ThemeDefinition:
    """Complete theme definition"""
    name: str
    mode: ThemeMode
    colors: ThemeColors
    typography: ThemeTypography
    spacing: ThemeSpacing
    is_high_contrast: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'mode': self.mode.value,
            'colors': self.colors.to_dict(),
            'typography': {
                'font_family': self.typography.font_family,
                'font_size_base': self.typography.font_size_base,
                'font_size_small': self.typography.font_size_small,
                'font_size_large': self.typography.font_size_large,
                'font_size_xlarge': self.typography.font_size_xlarge,
                'line_height': self.typography.line_height,
                'letter_spacing': self.typography.letter_spacing
            },
            'spacing': {
                'unit': self.spacing.unit,
                'padding_small': self.spacing.padding_small,
                'padding_medium': self.spacing.padding_medium,
                'padding_large': self.spacing.padding_large,
                'padding_xlarge': self.spacing.padding_xlarge,
                'margin_small': self.spacing.margin_small,
                'margin_medium': self.spacing.margin_medium,
                'margin_large': self.spacing.margin_large,
                'margin_xlarge': self.spacing.margin_xlarge,
                'border_radius': self.spacing.border_radius,
                'border_radius_small': self.spacing.border_radius_small,
                'border_radius_large': self.spacing.border_radius_large
            },
            'is_high_contrast': self.is_high_contrast
        }


class ThemeManager(QObject):
    """Theme management system"""
    
    theme_changed = pyqtSignal(ThemeDefinition)
    
    def __init__(self):
        super().__init__()
        self.themes: Dict[str, ThemeDefinition] = {}
        self.current_theme: Optional[ThemeDefinition] = None
        self.custom_themes_path = Path.home() / ".visionsub" / "themes"
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default themes
        self._init_default_themes()
        
        # Load custom themes
        self._load_custom_themes()
        
        # Set default theme
        self.set_theme("light")

    def _init_default_themes(self):
        """Initialize default themes"""
        # Light theme
        light_colors = ThemeColors(
            primary=QColor("#2563eb"),
            secondary=QColor("#64748b"),
            accent=QColor("#f59e0b"),
            background=QColor("#ffffff"),
            surface=QColor("#f8fafc"),
            text_primary=QColor("#1e293b"),
            text_secondary=QColor("#64748b"),
            border=QColor("#e2e8f0"),
            success=QColor("#10b981"),
            warning=QColor("#f59e0b"),
            error=QColor("#ef4444"),
            info=QColor("#3b82f6")
        )
        
        self.themes["light"] = ThemeDefinition(
            name="Light",
            mode=ThemeMode.LIGHT,
            colors=light_colors,
            typography=ThemeTypography(),
            spacing=ThemeSpacing()
        )
        
        # Dark theme
        dark_colors = ThemeColors(
            primary=QColor("#3b82f6"),
            secondary=QColor("#94a3b8"),
            accent=QColor("#fbbf24"),
            background=QColor("#0f172a"),
            surface=QColor("#1e293b"),
            text_primary=QColor("#f1f5f9"),
            text_secondary=QColor("#94a3b8"),
            border=QColor("#334155"),
            success=QColor("#34d399"),
            warning=QColor("#fbbf24"),
            error=QColor("#f87171"),
            info=QColor("#60a5fa")
        )
        
        self.themes["dark"] = ThemeDefinition(
            name="Dark",
            mode=ThemeMode.DARK,
            colors=dark_colors,
            typography=ThemeTypography(),
            spacing=ThemeSpacing()
        )
        
        # High contrast theme
        hc_colors = ThemeColors(
            primary=QColor("#000000"),
            secondary=QColor("#000000"),
            accent=QColor("#ff6600"),
            background=QColor("#ffffff"),
            surface=QColor("#ffffff"),
            text_primary=QColor("#000000"),
            text_secondary=QColor("#000000"),
            border=QColor("#000000"),
            success=QColor("#008000"),
            warning=QColor("#ff6600"),
            error=QColor("#ff0000"),
            info=QColor("#0000ff")
        )
        
        self.themes["high_contrast"] = ThemeDefinition(
            name="High Contrast",
            mode=ThemeMode.HIGH_CONTRAST,
            colors=hc_colors,
            typography=ThemeTypography(font_size_base=16),
            spacing=ThemeSpacing(border_radius=0),
            is_high_contrast=True
        )

    def _load_custom_themes(self):
        """Load custom themes from file system"""
        try:
            self.custom_themes_path.mkdir(parents=True, exist_ok=True)
            
            for theme_file in self.custom_themes_path.glob("*.json"):
                try:
                    with open(theme_file, 'r', encoding='utf-8') as f:
                        theme_data = json.load(f)
                    
                    theme = self._dict_to_theme(theme_data)
                    self.themes[theme.name.lower()] = theme
                    
                except Exception as e:
                    self.logger.error(f"Failed to load theme {theme_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create themes directory: {e}")

    def _dict_to_theme(self, data: Dict[str, Any]) -> ThemeDefinition:
        """Convert dictionary to ThemeDefinition"""
        colors = ThemeColors(**{
            role.value: QColor(color_str)
            for role, color_str in data['colors'].items()
        })
        
        typography = ThemeTypography(**data['typography'])
        spacing = ThemeSpacing(**data['spacing'])
        
        return ThemeDefinition(
            name=data['name'],
            mode=ThemeMode(data['mode']),
            colors=colors,
            typography=typography,
            spacing=spacing,
            is_high_contrast=data.get('is_high_contrast', False)
        )

    def set_theme(self, theme_name: str):
        """Set current theme"""
        theme_name = theme_name.lower()
        
        if theme_name not in self.themes:
            self.logger.warning(f"Theme '{theme_name}' not found, using default")
            theme_name = "light"
        
        self.current_theme = self.themes[theme_name]
        self._apply_theme()
        self.theme_changed.emit(self.current_theme)
        
        self.logger.info(f"Theme changed to: {self.current_theme.name}")

    def _apply_theme(self):
        """Apply theme to application"""
        if not self.current_theme:
            return
        
        app = QApplication.instance()
        palette = app.palette()
        
        colors = self.current_theme.colors
        
        # Apply colors to palette
        palette.setColor(QPalette.ColorRole.Window, colors.background)
        palette.setColor(QPalette.ColorRole.WindowText, colors.text_primary)
        palette.setColor(QPalette.ColorRole.Base, colors.surface)
        palette.setColor(QPalette.ColorRole.AlternateBase, colors.background)
        palette.setColor(QPalette.ColorRole.ToolTipBase, colors.surface)
        palette.setColor(QPalette.ColorRole.ToolTipText, colors.text_primary)
        palette.setColor(QPalette.ColorRole.Text, colors.text_primary)
        palette.setColor(QPalette.ColorRole.Button, colors.surface)
        palette.setColor(QPalette.ColorRole.ButtonText, colors.text_primary)
        palette.setColor(QPalette.ColorRole.BrightText, colors.accent)
        palette.setColor(QPalette.ColorRole.Link, colors.primary)
        palette.setColor(QPalette.ColorRole.Highlight, colors.primary)
        palette.setColor(QPalette.ColorRole.HighlightedText, colors.background)
        
        app.setPalette(palette)
        
        # Set default font
        app.setFont(self.current_theme.typography.get_font())

    def get_theme(self) -> ThemeDefinition:
        """Get current theme"""
        return self.current_theme

    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return list(self.themes.keys())

    def save_custom_theme(self, theme: ThemeDefinition):
        """Save custom theme to file"""
        try:
            self.custom_themes_path.mkdir(parents=True, exist_ok=True)
            
            theme_file = self.custom_themes_path / f"{theme.name.lower().replace(' ', '_')}.json"
            
            with open(theme_file, 'w', encoding='utf-8') as f:
                json.dump(theme.to_dict(), f, indent=2)
            
            self.themes[theme.name.lower()] = theme
            self.logger.info(f"Saved custom theme: {theme.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save custom theme: {e}")

    def create_custom_theme(
        self, 
        name: str, 
        base_theme: str = "light",
        color_overrides: Optional[Dict[str, str]] = None
    ) -> ThemeDefinition:
        """Create custom theme based on existing theme"""
        if base_theme not in self.themes:
            raise ValueError(f"Base theme '{base_theme}' not found")
        
        base = self.themes[base_theme]
        
        # Create a copy of the base theme
        import copy
        new_theme = copy.deepcopy(base)
        new_theme.name = name
        
        # Apply color overrides
        if color_overrides:
            for role_str, color_str in color_overrides.items():
                try:
                    role = ColorRole(role_str)
                    setattr(new_theme.colors, role.value, QColor(color_str))
                except ValueError:
                    self.logger.warning(f"Unknown color role: {role_str}")
        
        return new_theme


class ResponsiveLayout(QLayout):
    """Responsive layout that adapts to screen size"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[QLayoutItem] = []
        self.breakpoints = {
            SizeVariant.MOBILE: 768,
            SizeVariant.TABLET: 1024,
            SizeVariant.DESKTOP: 1440
        }
        self.current_variant = SizeVariant.DESKTOP
        
    def addItem(self, item: QLayoutItem):
        """Add item to layout"""
        self.items.append(item)
        
    def count(self) -> int:
        """Get number of items"""
        return len(self.items)
        
    def itemAt(self, index: int) -> QLayoutItem:
        """Get item at index"""
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
        
    def takeAt(self, index: int) -> QLayoutItem:
        """Remove item at index"""
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None
        
    def sizeHint(self) -> QSize:
        """Get size hint"""
        return QSize(400, 300)
        
    def minimumSize(self) -> QSize:
        """Get minimum size"""
        return QSize(200, 150)
        
    def setGeometry(self, rect: QRect):
        """Set geometry and update layout"""
        super().setGeometry(rect)
        self._do_layout(rect)
        
    def _do_layout(self, rect: QRect):
        """Perform layout based on current size variant"""
        # Determine current size variant
        width = rect.width()
        
        if width < self.breakpoints[SizeVariant.MOBILE]:
            self.current_variant = SizeVariant.MOBILE
        elif width < self.breakpoints[SizeVariant.TABLET]:
            self.current_variant = SizeVariant.TABLET
        elif width < self.breakpoints[SizeVariant.DESKTOP]:
            self.current_variant = SizeVariant.DESKTOP
        else:
            self.current_variant = SizeVariant.LARGE
        
        # Layout items based on variant
        self._layout_items(rect)
        
    def _layout_items(self, rect: QRect):
        """Layout items based on current size variant"""
        if not self.items:
            return
            
        theme_manager = get_theme_manager()
        spacing = theme_manager.current_theme.spacing
        
        if self.current_variant == SizeVariant.MOBILE:
            # Mobile: Single column
            y = rect.y()
            for item in self.items:
                item.setGeometry(QRect(
                    rect.x(), y,
                    rect.width(), item.sizeHint().height()
                ))
                y += item.sizeHint().height() + spacing.margin_medium
                
        elif self.current_variant == SizeVariant.TABLET:
            # Tablet: Two columns
            col_width = rect.width() // 2
            left_y = rect.y()
            right_y = rect.y()
            
            for i, item in enumerate(self.items):
                if i % 2 == 0:
                    # Left column
                    item.setGeometry(QRect(
                        rect.x(), left_y,
                        col_width - spacing.margin_medium // 2,
                        item.sizeHint().height()
                    ))
                    left_y += item.sizeHint().height() + spacing.margin_medium
                else:
                    # Right column
                    item.setGeometry(QRect(
                        rect.x() + col_width + spacing.margin_medium // 2, right_y,
                        col_width - spacing.margin_medium // 2,
                        item.sizeHint().height()
                    ))
                    right_y += item.sizeHint().height() + spacing.margin_medium
                    
        else:
            # Desktop/Large: Flexible grid
            self._layout_grid(rect)
            
    def _layout_grid(self, rect: QRect):
        """Layout items in a flexible grid"""
        if not self.items:
            return
            
        theme_manager = get_theme_manager()
        spacing = theme_manager.current_theme.spacing
        
        # Calculate grid dimensions
        item_count = len(self.items)
        cols = min(4, max(2, item_count))
        rows = (item_count + cols - 1) // cols
        
        item_width = (rect.width() - (cols - 1) * spacing.margin_medium) // cols
        item_height = 100  # Default item height
        
        for i, item in enumerate(self.items):
            row = i // cols
            col = i % cols
            
            x = rect.x() + col * (item_width + spacing.margin_medium)
            y = rect.y() + row * (item_height + spacing.margin_medium)
            
            item.setGeometry(QRect(x, y, item_width, item_height))


class StyledWidget(QWidget):
    """Base widget with theme-aware styling"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self._setup_styles()
        
        # Connect to theme changes
        self.theme_manager.theme_changed.connect(self._on_theme_changed)
        
    def _setup_styles(self):
        """Setup initial styles"""
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._update_styles()
        
    def _update_styles(self):
        """Update widget styles based on current theme"""
        if not self.theme_manager.current_theme:
            return
            
        theme = self.theme_manager.current_theme
        colors = theme.colors
        spacing = theme.spacing
        
        # Base styling
        style = f"""
            QWidget {{
                background-color: {colors.surface.name()};
                color: {colors.text_primary.name()};
                border: 1px solid {colors.border.name()};
                border-radius: {spacing.border_radius}px;
                font-family: {theme.typography.font_family};
                font-size: {theme.typography.font_size_base}px;
            }}
        """
        
        self.setStyleSheet(style)
        
    def _on_theme_changed(self, theme: ThemeDefinition):
        """Handle theme change"""
        self._update_styles()


class Card(StyledWidget):
    """Modern card widget"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup card UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self.theme_manager.current_theme.spacing.padding_large,
            self.theme_manager.current_theme.spacing.padding_large,
            self.theme_manager.current_theme.spacing.padding_large,
            self.theme_manager.current_theme.spacing.padding_large
        )
        
        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
            layout.addWidget(self.title_label)
            
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        
    def add_widget(self, widget: QWidget):
        """Add widget to card content"""
        self.content_layout.addWidget(widget)
        
    def add_layout(self, layout: QLayout):
        """Add layout to card content"""
        self.content_layout.addLayout(layout)


class Button(StyledWidget):
    """Modern button with animations"""
    
    clicked = pyqtSignal()
    
    def __init__(self, text: str, button_type: str = "primary", parent=None):
        super().__init__(parent)
        self.text = text
        self.button_type = button_type
        self._is_hovered = False
        self._is_pressed = False
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup button UI"""
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            self.theme_manager.current_theme.spacing.padding_medium,
            self.theme_manager.current_theme.spacing.padding_small,
            self.theme_manager.current_theme.spacing.padding_medium,
            self.theme_manager.current_theme.spacing.padding_small
        )
        
        self.label = QLabel(self.text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
    def enterEvent(self, event):
        """Handle mouse enter"""
        self._is_hovered = True
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave"""
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = True
            self.update()
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = False
            self.update()
            self.clicked.emit()
        super().mouseReleaseEvent(event)
        
    def paintEvent(self, event):
        """Custom paint event"""
        super().paintEvent(event)
        
        if not self.theme_manager.current_theme:
            return
            
        theme = self.theme_manager.current_theme
        colors = theme.colors
        
        # Get button color based on type
        color_map = {
            "primary": colors.primary,
            "secondary": colors.secondary,
            "accent": colors.accent,
            "success": colors.success,
            "warning": colors.warning,
            "error": colors.error
        }
        
        base_color = color_map.get(self.button_type, colors.primary)
        
        # Adjust color based on state
        if self._is_pressed:
            color = base_color.darker(120)
        elif self._is_hovered:
            color = base_color.lighter(110)
        else:
            color = base_color
            
        # Draw button background
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        rect = self.rect()
        painter.drawRoundedRect(
            rect.adjusted(1, 1, -1, -1),
            theme.spacing.border_radius,
            theme.spacing.border_radius
        )
        
        # Set text color
        text_color = colors.background if self.button_type != "secondary" else colors.text_primary
        self.label.setStyleSheet(f"color: {text_color.name()};")


# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get global theme manager instance"""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def initialize_theme_manager():
    """Initialize global theme manager"""
    global _theme_manager
    _theme_manager = ThemeManager()
    return _theme_manager


# Import required modules
from PyQt6.QtWidgets import QLabel
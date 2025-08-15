# Enhanced UI Components for VisionSub

## Overview

This directory contains enhanced UI components for the VisionSub video OCR application with modern design principles, improved user experience, and comprehensive security features.

## Key Features

### 🎨 Modern UI/UX Design
- **Theme System**: Dynamic theming with light, dark, and high-contrast modes
- **Responsive Layout**: Adaptive layouts for different screen sizes
- **Smooth Animations**: Professional transitions and micro-interactions
- **Material Design**: Modern card-based interface with proper spacing and typography
- **Accessibility**: Full keyboard navigation and screen reader support

### 🔒 Security Features
- **Input Validation**: Comprehensive sanitization of all user inputs
- **File Security**: Secure file handling with size and format validation
- **Path Sanitization**: Protection against directory traversal attacks
- **Content Filtering**: Malicious content detection and filtering
- **Secure Rendering**: Safe display of OCR results and user-generated content

### ⚡ Performance Optimizations
- **Efficient Rendering**: Optimized paint events and memory management
- **Lazy Loading**: Dynamic content loading for large datasets
- **Caching**: Intelligent caching of UI components and data
- **Background Processing**: Non-blocking operations for better responsiveness

## Components

### 1. Enhanced Main Window (`enhanced_main_window.py`)

**Features:**
- Modern toolbar with contextual actions
- Secure file dialogs with validation
- Enhanced status bar with performance indicators
- Theme-aware styling and animations
- Comprehensive keyboard shortcuts

**Security:**
- Secure file path validation
- Input sanitization for all fields
- Protection against malicious file uploads
- Safe configuration management

```python
# Usage example
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.view_models.main_view_model import MainViewModel

vm = MainViewModel()
window = EnhancedMainWindow(vm)
window.show()
```

### 2. Enhanced Video Player (`enhanced_video_player.py`)

**Features:**
- Modern video controls with smooth animations
- Zoom and pan functionality
- ROI selection with visual feedback
- Performance overlay and statistics
- Keyboard shortcuts for navigation

**Security:**
- Frame size validation to prevent memory issues
- Secure coordinate mapping
- Protection against malicious video content
- Safe zoom level limits

```python
# Usage example
from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer

player = EnhancedVideoPlayer()
player.update_frame(frame_array)
player.show()
```

### 3. Enhanced Settings Dialog (`enhanced_settings_dialog.py`)

**Features:**
- Tabbed interface for organized settings
- Real-time validation and feedback
- Theme preview functionality
- Security status indicators
- Export/import configuration

**Security:**
- Input validation for all settings
- Secure configuration storage
- Protection against configuration injection
- Safe file handling for exports

```python
# Usage example
from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
from visionsub.models.config import AppConfig

config = AppConfig()
dialog = EnhancedSettingsDialog(config)
dialog.config_changed.connect(lambda c: print(f"Config updated: {c}"))
dialog.show()
```

### 4. Enhanced OCR Preview (`enhanced_ocr_preview.py`)

**Features:**
- Real-time text highlighting based on confidence
- Advanced filtering and search capabilities
- Export functionality with multiple formats
- Statistics and analytics display
- Secure text rendering

**Security:**
- Text sanitization for display
- Protection against XSS attacks
- Safe content filtering
- Memory usage limits

```python
# Usage example
from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview, OCRResult

result = OCRResult(
    text="Sample text",
    confidence=0.95,
    language="en",
    position=QRect(0, 0, 100, 30),
    timestamp=1.0
)

preview = EnhancedOCRPreview()
preview.add_result(result)
preview.show()
```

### 5. Enhanced Subtitle Editor (`enhanced_subtitle_editor.py`)

**Features:**
- Advanced table model with sorting and filtering
- Real-time validation and error checking
- Undo/redo functionality
- Batch operations
- Import/export in multiple formats

**Security:**
- Subtitle content validation
- Time format validation
- Protection against malicious subtitle files
- Safe file handling

```python
# Usage example
from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
from visionsub.models.subtitle import SubtitleItem

subtitle = SubtitleItem(
    index=1,
    start_time="00:00:01,000",
    end_time="00:00:03,000",
    content="Sample subtitle"
)

editor = EnhancedSubtitleEditor()
editor.set_subtitles([subtitle])
editor.show()
```

### 6. Enhanced Main Application (`enhanced_main.py`)

**Features:**
- Modern application lifecycle management
- Splash screen with loading indicators
- System tray integration
- Automatic cleanup and optimization
- Comprehensive error handling

**Security:**
- Secure application initialization
- Protected configuration loading
- Safe cleanup procedures
- Memory management

```python
# Usage example
from visionsub.ui.enhanced_main import run_enhanced_gui

if __name__ == "__main__":
    sys.exit(run_enhanced_gui())
```

## Security Implementation

### Input Validation
All user inputs are validated using comprehensive validators:
```python
class SecureInputValidator(QValidator):
    def validate(self, input_str: str, pos: int):
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return (QValidator.State.Invalid, input_str, pos)
        
        # Additional validation logic
        return (QValidator.State.Acceptable, input_str, pos)
```

### File Security
Secure file handling with multiple validation layers:
```python
class SecureFileDialog(QFileDialog):
    def _validate_file(self, file_path: str) -> bool:
        # Check file existence and permissions
        # Validate file extension and size
        # Check for malicious path patterns
        # Verify file content safety
        return is_valid
```

### Content Sanitization
Comprehensive text sanitization:
```python
class SecureTextRenderer:
    def sanitize_text(self, text: str) -> str:
        # Remove control characters
        # Filter dangerous characters
        # Limit text length
        # Encode special characters safely
        return sanitized_text
```

## Theme System

### Available Themes
- **Light**: Clean, bright interface suitable for well-lit environments
- **Dark**: Reduced eye strain for low-light conditions
- **High Contrast**: Maximum accessibility for visually impaired users
- **Custom**: User-defined themes with full customization

### Theme Usage
```python
# Get theme manager
theme_manager = get_theme_manager()

# Set theme
theme_manager.set_theme("dark")

# Create custom theme
custom_theme = theme_manager.create_custom_theme(
    name="My Theme",
    base_theme="dark",
    color_overrides={
        "primary": "#FF6B6B",
        "accent": "#4ECDC4"
    }
)
```

## Performance Optimizations

### Memory Management
- Automatic cleanup of unused resources
- Memory usage monitoring and optimization
- Efficient data structures for large datasets
- Lazy loading of components

### Rendering Optimizations
- Optimized paint events with minimal redraws
- Hardware acceleration where available
- Efficient use of Qt's graphics system
- Caching of frequently used resources

### Background Processing
- Non-blocking operations for better responsiveness
- Worker threads for intensive tasks
- Progress indicators for long operations
- Cancelable operations

## Integration Guide

### Basic Integration
1. Import required components
2. Initialize theme system
3. Create main window with view model
4. Connect signals and slots
5. Run application

```python
import sys
from PyQt6.QtWidgets import QApplication
from visionsub.ui.enhanced_main import run_enhanced_gui

if __name__ == "__main__":
    sys.exit(run_enhanced_gui())
```

### Advanced Integration
For custom applications:
```python
from visionsub.ui.theme_system import initialize_theme_manager
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.view_models.main_view_model import MainViewModel

# Initialize theme system
theme_manager = initialize_theme_manager()

# Create view model
vm = MainViewModel()

# Create main window
window = EnhancedMainWindow(vm)

# Customize as needed
window.setWindowTitle("My Custom App")
window.resize(1200, 800)

# Show and run
window.show()
sys.exit(app.exec())
```

## Best Practices

### Security
1. Always validate user inputs
2. Use secure file dialogs
3. Sanitize all displayed content
4. Implement proper error handling
5. Follow principle of least privilege

### Performance
1. Use lazy loading for large datasets
2. Implement proper caching strategies
3. Optimize paint events
4. Use background processing
5. Monitor memory usage

### User Experience
1. Provide clear visual feedback
2. Implement consistent interactions
3. Support keyboard navigation
4. Include accessibility features
5. Offer customization options

## Testing

### Unit Tests
Run component tests:
```bash
python -m pytest tests/ui/test_enhanced_components.py
```

### Integration Tests
Test full application:
```bash
python -m pytest tests/ui/test_enhanced_integration.py
```

### Security Tests
Validate security features:
```bash
python -m pytest tests/ui/test_security_features.py
```

## Contributing

### Development Setup
1. Install development dependencies
2. Set up pre-commit hooks
3. Configure development environment
4. Run tests before committing

### Code Style
- Follow PEP 8 guidelines
- Use type hints consistently
- Write comprehensive docstrings
- Include security considerations

### Security Considerations
- Validate all inputs
- Sanitize all outputs
- Use secure coding practices
- Regular security reviews

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- Create GitHub issues for bugs and feature requests
- Check documentation for troubleshooting
- Review security guidelines for safe usage
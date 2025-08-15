# VisionSub 测试框架

这是为 VisionSub 视频OCR工具设计的全面测试框架，包含以下测试组件：

## 测试组件

### 1. 增强UI测试 (`test_enhanced_ui.py`)
- **视频播放器测试**: 视频加载、播放控制、帧提取、快捷键
- **OCR预览测试**: 图像加载、OCR处理、文本编辑、缩放控制
- **主题系统测试**: 主题切换、自定义主题、主题持久化
- **设置对话框测试**: OCR设置、处理设置、UI设置、验证
- **ROI选择测试**: ROI选择、预设大小、调整、验证
- **字幕编辑器测试**: 字幕编辑、时间调整、搜索、导出格式
- **主窗口测试**: 菜单操作、工具栏、状态持久化、最近文件

### 2. 后端服务测试 (`test_enhanced_backend.py`)
- **视频处理测试**: 视频信息提取、帧提取、批处理、场景检测
- **OCR处理测试**: OCR处理、ROI处理、置信度过滤、语言检测
- **字幕处理测试**: 字幕创建、时间调整、合并拆分、格式转换
- **配置管理测试**: 配置保存加载、验证、合并、重置
- **缓存管理测试**: 缓存存储检索、过期、大小限制、持久化
- **验证测试**: 视频文件验证、配置验证、OCR结果验证、字幕验证

### 3. 安全测试 (`test_security.py`)
- **输入验证测试**: 路径遍历、文件扩展名、文件大小、配置注入
- **文件上传安全**: 文件类型验证、文件名清理、目录遍历防护
- **认证授权测试**: 会话安全、权限验证、访问控制、速率限制
- **数据保护测试**: 敏感数据处理、数据加密、日志清理
- **漏洞扫描测试**: 依赖漏洞、代码安全分析、不安全配置、硬编码密钥
- **渗透测试测试**: XSS漏洞、CSRF防护、SSRF防护、RCE防护
- **安全头测试**: 安全头实现、CSP配置

### 4. 性能测试 (`test_performance.py`)
- **性能指标测试**: CPU使用率、内存使用率、磁盘使用率、网络监控
- **视频处理性能**: 视频信息提取、帧提取、批处理、场景检测
- **OCR处理性能**: 单图像OCR、批处理OCR、并发处理、内存使用
- **缓存性能**: 缓存写入、缓存读取、并发访问、内存使用
- **负载测试**: 并发视频处理、高频操作、内存泄漏检测
- **扩展性测试**: 线性扩展、并发扩展
- **资源利用测试**: CPU利用率、内存利用率

### 5. 集成测试 (`test_integration.py`)
- **UI组件集成测试**: 主窗口初始化、视频OCR集成、OCR字幕集成、设置集成
- **后端组件集成测试**: 视频OCR集成、OCR字幕集成、配置集成、缓存集成
- **事件处理集成测试**: 视频加载事件、OCR完成事件、字幕更改事件、设置更改事件
- **工作流集成测试**: 完整工作流、错误处理工作流、状态持久化工作流
- **异步集成测试**: 异步视频处理、异步OCR处理、并发处理

### 6. 端到端测试 (`test_e2e.py`)
- **完整工作流测试**: 视频处理工作流、批处理工作流、设置自定义工作流
- **错误恢复测试**: 错误处理工作流、配置错误恢复
- **用户验收测试**: 首次用户体验、有经验用户工作流、高级用户功能
- **性能验收测试**: 性能标准验证
- **无障碍测试**: 键盘导航、屏幕阅读器、高对比度、字体大小、色盲支持
- **跨平台测试**: 路径处理、平台特定功能
- **数据完整性测试**: 视频数据完整性、OCR数据完整性、字幕数据完整性、配置数据完整性

### 7. 测试工具 (`test_utils/`)
- **测试数据生成**: 图像生成、视频生成、OCR结果生成、字幕生成
- **模拟对象**: 模拟OCR处理器、模拟视频处理器
- **性能监控**: 性能测量、系统信息监控
- **测试环境**: 临时文件管理、临时目录管理
- **测试助手**: 信号等待、图像比较、测试数据创建
- **工厂模式**: OCR结果工厂、字幕条目工厂、配置工厂

## 测试配置

### pytest配置
```python
# 测试标记
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "performance: marks tests as performance tests",
    "gui: marks tests as GUI tests",
    "security: marks tests as security tests",
    "e2e: marks tests as end-to-end tests",
    "load: marks tests as load tests",
    "accessibility: marks tests as accessibility tests",
    "regression: marks tests as regression tests",
    "smoke: marks tests as smoke tests"
]

# 覆盖率配置
[tool.coverage.run]
source = ["src/visionsub"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/env/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod"
]
```

### 依赖项
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-qt = "^4.2.0"
pytest-mock = "^3.12.0"
pytest-asyncio = "^0.21.1"
pytest-benchmark = "^4.0.0"
pytest-xdist = "^3.5.0"
pytest-randomly = "^3.15.0"
pytest-html = "^4.1.1"
pytest-selenium = "^4.1.0"
pytest-cov = "^4.1.0"
locust = "^2.17.0"
bandit = "^1.7.5"
safety = "^2.3.5"
schemathesis = "^3.23.0"
factory-boy = "^3.3.0"
faker = "^19.12.0"
responses = "^0.24.1"
freezegun = "^1.2.2"
vcrpy = "^5.1.0"
allure-pytest = "^2.13.2"
```

## 使用方法

### 运行所有测试
```bash
pytest
```

### 运行特定测试类别
```bash
# 单元测试
pytest -m unit

# 集成测试
pytest -m integration

# 性能测试
pytest -m performance

# 安全测试
pytest -m security

# 端到端测试
pytest -m e2e

# UI测试
pytest -m gui
```

### 运行特定测试文件
```bash
# UI测试
pytest test_enhanced_ui.py -v

# 后端测试
pytest test_enhanced_backend.py -v

# 安全测试
pytest test_security.py -v

# 性能测试
pytest test_performance.py -v

# 集成测试
pytest test_integration.py -v

# 端到端测试
pytest test_e2e.py -v
```

### 带覆盖率报告
```bash
pytest --cov=src/visionsub --cov-report=html --cov-report=term-missing
```

### 并行运行测试
```bash
pytest -n auto  # 自动检测CPU核心数
pytest -n 4    # 使用4个进程
```

### 生成HTML报告
```bash
pytest --html=test_reports/report.html
```

### 性能基准测试
```bash
pytest --benchmark-only
pytest --benchmark-only --benchmark-sort=mean
```

## 测试最佳实践

### 1. 测试命名
- 使用描述性的测试名称
- 遵循 `test_` 前缀约定
- 测试类使用 `Test` 前缀

### 2. 测试组织
- 按功能模块组织测试
- 使用fixtures进行共享设置
- 使用标记进行测试分类

### 3. 测试数据
- 使用工厂模式生成测试数据
- 避免硬编码测试数据
- 清理测试数据

### 4. 断言
- 使用明确的断言消息
- 测试预期行为和边界条件
- 测试错误情况

### 5. 模拟和存根
- 模拟外部依赖
- 使用pytest-mock进行模拟
- 避免过度模拟

### 6. 性能考虑
- 标记慢速测试
- 使用性能基准测试
- 监控内存使用

## 持续集成

### GitHub Actions配置
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install
    
    - name: Run tests
      run: |
        poetry run pytest -m "not slow and not gui" --cov=src/visionsub --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

## 测试报告

测试框架会生成以下报告：
- **覆盖率报告**: `htmlcov/index.html`
- **HTML测试报告**: `test_reports/report.html`
- **性能基准报告**: 控制台输出
- **安全扫描报告**: 控制台输出

## 故障排除

### 常见问题
1. **Qt测试失败**: 确保安装了pytest-qt
2. **内存不足**: 减少并发测试数量
3. **权限问题**: 确保测试目录有写权限
4. **依赖问题**: 运行 `poetry install`

### 调试技巧
- 使用 `-v` 参数获取详细输出
- 使用 `-s` 参数禁用输出捕获
- 使用 `--pdb` 在失败时启动调试器
- 使用 `--lf` 只运行失败的测试

这个测试框架为VisionSub应用提供了全面的质量保证，确保应用的稳定性、安全性和性能。
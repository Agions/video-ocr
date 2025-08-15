#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频OCR字幕提取工具 - 上传界面
参考 SubtitleOCR 实现的视频上传及字幕提取界面
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 全局变量定义
# TkinterDnD相关变量将在导入时设置
TK_DND_AVAILABLE = False
DND_FILES = None

# 先尝试确保依赖已安装
def ensure_dependency(package_name, import_name=None, install_cmd=None):
    """尝试导入依赖，如果失败则尝试安装"""
    if import_name is None:
        import_name = package_name

    if install_cmd is None:
        install_cmd = f"pip install {package_name}"

    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"未找到 {import_name} 模块，尝试安装...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} 安装成功！")
            return True
        except Exception as e:
            print(f"安装 {package_name} 失败: {e}")
            print(f"请手动运行: {install_cmd}")
            return False

# 尝试安装基本依赖
ensure_dependency("pillow", "PIL")
ensure_dependency("opencv-python", "cv2")
ensure_dependency("tqdm")
ensure_dependency("pytesseract")

# PIL导入错误处理
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: 未找到PIL库，图像预览功能将不可用。尝试运行 'pip install pillow' 来安装。")

# OpenCV导入错误处理
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: 未找到OpenCV库，视频处理功能将不可用。尝试运行 'pip install opencv-python' 来安装。")

# tqdm导入错误处理
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: 未找到tqdm库，进度条功能将不可用。尝试运行 'pip install tqdm' 来安装。")

# Tesseract OCR导入错误处理
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print(f"Tesseract版本: {pytesseract.get_tesseract_version()}")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("\n警告: 未找到pytesseract库，OCR功能将不可用。")
    print("请运行 'pip install pytesseract' 安装Python库。")
    print("并确保系统已安装Tesseract OCR:")
    print("  - macOS: brew install tesseract tesseract-lang")
    print("  - Ubuntu: sudo apt-get install tesseract-ocr")
    print("  - Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装")
except Exception as e:
    TESSERACT_AVAILABLE = False
    print(f"\n警告: Tesseract可能未正确安装: {e}")
    print("请确保系统已安装Tesseract OCR引擎")

# 尝试导入tkinterdnd2（拖放支持，可选）
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TK_DND_AVAILABLE = True
except ImportError:
    TK_DND_AVAILABLE = False
    print("警告: 未找到tkinterdnd2模块，拖放功能将不可用。尝试运行 'pip install tkinterdnd2' 来安装。")

# 导入主要模块（如果出错，给出更明确的提示）
try:
    # 如果tqdm模块无法导入，提供一个简单的替代
    if not TQDM_AVAILABLE:
        class SimpleTqdm:
            def __init__(self, iterable=None, total=None, **kwargs):
                self.iterable = iterable
                self.total = total or (len(iterable) if iterable is not None else None)
                self.current = 0

            def update(self, n=1):
                self.current += n
                if self.total:
                    progress = self.current / self.total * 100
                    print(f"进度: {progress:.1f}%", end="\r")

            def close(self):
                print()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def __iter__(self):
                if self.iterable is None:
                    return range(self.total).__iter__()
                for item in self.iterable:
                    yield item
                    self.update()

        # 替换tqdm模块
        sys.modules['tqdm'] = type('', (), {'tqdm': SimpleTqdm})

    # 导入前检查关键依赖是否可用
    if not TESSERACT_AVAILABLE:
        print("\n警告: 缺少OCR核心依赖，程序可能无法正常工作")

        # 询问用户是否继续
        if not os.environ.get('IGNORE_TESSERACT_ERROR'):
            if input("是否仍然尝试启动程序？(y/n): ").lower() != 'y':
                print("程序退出。请先安装必要的依赖。")
                sys.exit(1)
            os.environ['IGNORE_TESSERACT_ERROR'] = '1'  # 设置环境变量避免重复询问

    # 导入主模块
    from main import VideoOCR
except ImportError as e:
    error_msg = f"错误: 无法导入主模块: {e}"
    print(error_msg)

    # 提供更详细的错误指导
    if "tesseract" in str(e).lower():
        print("\n这是一个与OCR引擎相关的错误，请安装Tesseract OCR:")
        print("1. 运行 'bash install_deps.sh' 安装所有依赖")
        print("2. 确保系统已安装Tesseract OCR引擎")
    else:
        print("请确保所有依赖已正确安装。运行 'bash install_deps.sh' 或 'pip install -r requirements.txt'")

    # 显示对话框，如果在GUI环境下
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("依赖错误", error_msg + "\n\n请运行 'bash install_deps.sh' 安装所有依赖")
    except Exception:
        pass

    sys.exit(1)


class DragDropFrame(ttk.Frame):
    """支持拖放的Frame"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # 拖放区样式
        self.configure(style="DnD.TFrame")

        # 标签
        self.header_label = ttk.Label(
            self,
            text="拖放视频文件到此处或点击选择文件",
            font=("Helvetica", 14)
        )
        self.header_label.pack(pady=(20, 10))

        # 图标框架
        self.icon_frame = ttk.Frame(self)
        self.icon_frame.pack(pady=10)

        # 加载默认图标
        self._load_default_icon()

        # 文件名标签
        self.filename_label = ttk.Label(
            self,
            text="未选择文件",
            font=("Helvetica", 10),
            wraplength=350
        )
        self.filename_label.pack(pady=(10, 20))

        # 回调函数
        self.on_file_selected_callback = None

        # 绑定点击事件
        self.bind("<Button-1>", self.on_click)
        self.header_label.bind("<Button-1>", self.on_click)
        self.icon_frame.bind("<Button-1>", self.on_click)
        self.filename_label.bind("<Button-1>", self.on_click)

        # 如果拖放可用，绑定拖放事件
        if TK_DND_AVAILABLE:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self.on_drop)

    def _load_default_icon(self):
        """加载默认图标"""
        if PIL_AVAILABLE:
            # 尝试创建简单的视频图标
            try:
                icon_size = 64
                icon = Image.new('RGBA', (icon_size, icon_size), (240, 240, 240, 0))

                # 使用PhotoImage显示图标
                self.icon_image = ImageTk.PhotoImage(icon)
                self.icon_label = ttk.Label(self.icon_frame, image=self.icon_image)
                self.icon_label.pack()

                # 绑定点击事件
                self.icon_label.bind("<Button-1>", self.on_click)
            except Exception as e:
                print(f"创建默认图标时出错: {e}")
                self.icon_label = ttk.Label(self.icon_frame, text="📺")
                self.icon_label.pack()
        else:
            # 如果PIL不可用，显示文本图标
            self.icon_label = ttk.Label(self.icon_frame, text="📺", font=("Helvetica", 24))
            self.icon_label.pack()

    def on_click(self, event):
        """点击时打开文件选择对话框"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.filename_label.config(text=os.path.basename(file_path))

            # 调用回调
            if self.on_file_selected_callback:
                self.on_file_selected_callback(file_path)

    def on_drop(self, event):
        """拖放文件时触发"""
        file_path = event.data

        # 修正Windows路径格式和引号问题
        file_path = file_path.replace("{", "").replace("}", "")
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        if os.path.isfile(file_path):
            self.filename_label.config(text=os.path.basename(file_path))

            # 调用回调
            if self.on_file_selected_callback:
                self.on_file_selected_callback(file_path)
        else:
            messagebox.showerror("错误", "请选择有效的视频文件")

    def set_on_file_selected(self, callback):
        """设置文件选择回调"""
        self.on_file_selected_callback = callback


class VideoPreview(ttk.Frame):
    """视频预览框架"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # 预览标题
        self.title_label = ttk.Label(
            self,
            text="视频预览",
            font=("Helvetica", 12, "bold")
        )
        self.title_label.pack(pady=(10, 5))

        # 预览框架
        self.preview_frame = ttk.Frame(self, borderwidth=1, relief="solid")
        self.preview_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # 预览标签
        self.preview_label = ttk.Label(
            self.preview_frame,
            text="载入视频后显示预览...",
            font=("Helvetica", 10)
        )
        self.preview_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def set_video(self, video_path):
        """设置视频并显示第一帧"""
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            self.preview_label.config(text="需要安装OpenCV和PIL才能显示预览")
            return

        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.preview_label.config(text="无法打开视频文件")
                return

            # 读取第一帧
            ret, frame = cap.read()
            if not ret:
                self.preview_label.config(text="无法读取视频帧")
                cap.release()
                return

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            # 格式化时长显示
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # 调整大小以适应预览区域
            max_width = 320
            max_height = 180
            h, w = frame.shape[:2]

            # 计算宽高比
            ratio = min(max_width / w, max_height / h)
            new_size = (int(w * ratio), int(h * ratio))

            # 调整图像大小
            resized = cv2.resize(frame, new_size)

            # 转换颜色空间从BGR到RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # 创建PIL图像
            image = Image.fromarray(rgb_frame)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image)

            # 更新预览
            self.preview_label.config(text="", image=photo)
            self.preview_label.image = photo  # 保持引用

            # 添加视频信息标签
            info_text = f"分辨率: {w}x{h} | 帧率: {fps:.2f} fps | 时长: {duration_str}"
            info_label = ttk.Label(
                self,
                text=info_text,
                font=("Helvetica", 9)
            )

            # 移除之前的信息标签（如果有）
            for widget in self.winfo_children():
                if widget != self.title_label and widget != self.preview_frame:
                    widget.destroy()

            info_label.pack(pady=(5, 10))

            # 释放资源
            cap.release()

        except Exception as e:
            self.preview_label.config(text=f"预览生成失败: {str(e)}")


class OCROptionsFrame(ttk.LabelFrame):
    """OCR选项框架"""

    def __init__(self, master, **kwargs):
        super().__init__(master, text="OCR选项", **kwargs)

        # 语言选择
        self.lang_frame = ttk.Frame(self)
        self.lang_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.lang_frame, text="识别语言:").pack(side=tk.LEFT, padx=(0, 5))

        self.lang_var = tk.StringVar(value="chi_sim")
        languages = [
            ("简体中文", "chi_sim"),
            ("繁体中文", "chi_tra"),
            ("英文", "eng"),
            ("日文", "jpn"),
            ("韩文", "kor"),
            ("简体中文+英文", "chi_sim+eng")
        ]

        self.lang_combobox = ttk.Combobox(
            self.lang_frame,
            textvariable=self.lang_var,
            values=[lang[0] for lang in languages],
            state="readonly",
            width=15
        )
        self.lang_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lang_combobox.current(0)

        # 语言代码映射
        self.lang_codes = {lang[0]: lang[1] for lang in languages}

        # 字幕区域选择
        self.area_frame = ttk.Frame(self)
        self.area_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.area_frame, text="字幕区域:").pack(side=tk.LEFT, padx=(0, 5))

        self.area_var = tk.StringVar(value="下部")
        areas = [
            ("下部(70%-100%)", "下部"),
            ("上部(0%-30%)", "上部"),
            ("全屏(0%-100%)", "全屏"),
            ("中部(30%-70%)", "中部")
        ]

        self.area_combobox = ttk.Combobox(
            self.area_frame,
            textvariable=self.area_var,
            values=[area[0] for area in areas],
            state="readonly",
            width=15
        )
        self.area_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.area_combobox.current(0)

        # 区域映射
        self.area_ranges = {
            "下部": (0.7, 1.0),
            "上部": (0.0, 0.3),
            "全屏": (0.0, 1.0),
            "中部": (0.3, 0.7)
        }

        # 提取间隔
        self.interval_frame = ttk.Frame(self)
        self.interval_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.interval_frame, text="提取间隔(秒):").pack(side=tk.LEFT, padx=(0, 5))

        self.interval_var = tk.DoubleVar(value=1.0)
        self.interval_scale = ttk.Scale(
            self.interval_frame,
            from_=0.1,
            to=5.0,
            variable=self.interval_var,
            orient=tk.HORIZONTAL
        )
        self.interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.interval_label = ttk.Label(self.interval_frame, text="1.0")
        self.interval_label.pack(side=tk.LEFT, padx=(0, 5))

        # 绑定事件
        self.interval_scale.bind("<Motion>", self._update_interval_label)
        self.interval_var.trace_add("write", self._update_interval_label_from_var)

    def _update_interval_label(self, event=None):
        self.interval_label.config(text=f"{self.interval_var.get():.1f}")

    def _update_interval_label_from_var(self, *args):
        self.interval_label.config(text=f"{self.interval_var.get():.1f}")

    def get_options(self):
        """获取OCR选项"""
        lang_name = self.lang_var.get()
        lang_code = self.lang_codes.get(lang_name, "chi_sim")

        area_name = self.area_var.get().split('(')[0]
        subtitle_area = self.area_ranges.get(area_name, (0.7, 1.0))

        interval = self.interval_var.get()

        return {
            "lang": lang_code,
            "subtitle_area": subtitle_area,
            "interval": interval
        }


class VideoOCRUploadApp:
    """视频OCR上传应用"""

    def __init__(self, root):
        """初始化应用"""
        self.root = root
        self.root.title("视频OCR字幕提取工具")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # 设置主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 设置DPI缩放
        self.check_dpi_scaling()

        # 检查OCR引擎
        if not TESSERACT_AVAILABLE:
            self.show_dependency_warning()

        # 设置主题
        self.setup_theme()

        # 设置UI组件
        self.setup_ui()

        # 当前选中的视频文件
        self.current_video = None

        # 正在处理的标志
        self.processing = False

    def show_dependency_warning(self):
        """显示依赖警告"""
        if not TESSERACT_AVAILABLE:
            warning = tk.Toplevel(self.root)
            warning.title("缺少OCR引擎")
            warning.geometry("450x250")
            warning.transient(self.root)
            warning.grab_set()

            ttk.Label(
                warning,
                text="警告: 未检测到Tesseract OCR引擎",
                font=("Helvetica", 12, "bold")
            ).pack(pady=(20, 10))

            ttk.Label(
                warning,
                text="OCR功能将不可用。请安装以下组件:",
                font=("Helvetica", 10)
            ).pack(pady=(0, 10))

            ttk.Label(
                warning,
                text="1. 安装pytesseract: pip install pytesseract\n\n"
                     "2. 安装Tesseract OCR引擎:\n"
                     "   - macOS: brew install tesseract tesseract-lang\n"
                     "   - Ubuntu: sudo apt install tesseract-ocr\n"
                     "   - Windows: 从Github下载安装程序",
                justify=tk.LEFT
            ).pack(padx=20)

            ttk.Button(
                warning,
                text="我知道了",
                command=warning.destroy
            ).pack(pady=20)

    def check_dpi_scaling(self):
        """检查并设置DPI缩放"""
        try:
            # Windows系统下的DPI设置
            if os.name == 'nt':
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    def setup_theme(self):
        """设置应用主题"""
        style = ttk.Style()

        # 设置主题
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")

        # 自定义风格
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5")
        style.configure("TButton", padding=5)
        style.configure("DnD.TFrame", borderwidth=2, relief="groove", background="#f0f0f0")

    def setup_ui(self):
        """设置用户界面"""
        # 创建上下两部分布局
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X)

        # 上部分：分为左右两栏
        self.left_frame = ttk.Frame(self.top_frame, width=400)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.right_frame = ttk.Frame(self.top_frame, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 左侧：拖放上传区
        self.upload_frame = DragDropFrame(self.left_frame)
        self.upload_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.upload_frame.set_on_file_selected(self.on_video_selected)

        # 右侧：预览和选项
        self.preview_frame = VideoPreview(self.right_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.options_frame = OCROptionsFrame(self.right_frame)
        self.options_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        # 底部：输出目录和开始按钮
        ttk.Label(self.bottom_frame, text="输出目录:").pack(side=tk.LEFT, padx=(10, 5))

        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.bottom_frame, textvariable=self.output_var, width=50)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # 设置默认输出路径
        default_output = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(default_output):
            self.output_var.set(default_output)
        else:
            self.output_var.set(os.getcwd())

        self.browse_button = ttk.Button(
            self.bottom_frame,
            text="浏览...",
            command=self.browse_output
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)

        self.extract_button = ttk.Button(
            self.bottom_frame,
            text="开始提取",
            command=self.start_extraction,
            style="Accent.TButton"
        )
        self.extract_button.pack(side=tk.LEFT, padx=10)

        # 设置提取按钮样式
        style = ttk.Style()
        style.configure("Accent.TButton", background="#4CAF50", foreground="white")

        # 状态栏
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(
            self.status_frame,
            text="就绪",
            font=("Helvetica", 9),
            foreground="#555555"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # 进度条（初始隐藏）
        self.progress = ttk.Progressbar(
            self.status_frame,
            mode="indeterminate",
            length=200
        )

    def on_video_selected(self, file_path):
        """视频文件选择回调"""
        self.current_video = file_path
        self.status_label.config(text=f"已选择: {os.path.basename(file_path)}")

        # 更新预览
        self.preview_frame.set_video(file_path)

        # 设置默认输出目录为视频所在目录
        video_dir = os.path.dirname(file_path)
        if os.path.exists(video_dir):
            self.output_var.set(video_dir)

    def browse_output(self):
        """浏览并选择输出目录"""
        directory = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=self.output_var.get()
        )

        if directory:
            self.output_var.set(directory)

    def start_extraction(self):
        """开始提取字幕"""
        if self.processing:
            return

        if not self.current_video:
            messagebox.showwarning("警告", "请先选择视频文件")
            return

        if not TESSERACT_AVAILABLE:
            messagebox.showerror("错误", "未检测到Tesseract OCR引擎，无法进行提取。请安装Tesseract OCR。")
            return

        output_dir = self.output_var.get()
        if not os.path.exists(output_dir):
            messagebox.showwarning("警告", "输出目录不存在")
            return

        # 获取OCR选项
        options = self.options_frame.get_options()

        # 设置进度条
        self.progress.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)
        self.progress.start(10)

        # 更新状态
        self.status_label.config(text="正在处理视频...", foreground="#1976D2")

        # 禁用UI
        self.disable_ui()

        # 开始提取线程
        self.processing = True
        extraction_thread = threading.Thread(
            target=self.process_extraction,
            args=(options,)
        )
        extraction_thread.daemon = True
        extraction_thread.start()

    def process_extraction(self, options):
        """处理提取过程的线程"""
        try:
            # 提取输出文件名
            video_name = os.path.basename(self.current_video)
            video_name_no_ext = os.path.splitext(video_name)[0]
            output_file = os.path.join(self.output_var.get(), f"{video_name_no_ext}.srt")

            # 创建OCR引擎实例
            ocr = VideoOCR(
                video_path=self.current_video,
                subtitle_area=options["subtitle_area"],
                lang=options["lang"],
                interval=options["interval"]
            )

            # 提取字幕
            result = ocr.extract_subtitles(output_file)

            # 更新UI
            self.root.after(0, lambda: self.extraction_complete(True, f"提取完成，输出文件: {output_file}"))

        except Exception as e:
            # 出现错误
            error_msg = str(e)
            self.root.after(0, lambda: self.extraction_complete(False, f"提取失败: {error_msg}"))

        finally:
            # 重置处理标志
            self.processing = False

    def extraction_complete(self, success, message):
        """提取完成后的UI更新"""
        # 停止进度条
        self.progress.stop()
        self.progress.pack_forget()

        # 更新状态
        if success:
            self.status_label.config(text=message, foreground="#4CAF50")
            messagebox.showinfo("完成", message)
        else:
            self.status_label.config(text=message, foreground="#F44336")
            messagebox.showerror("错误", message)

        # 启用UI
        self.enable_ui()

    def disable_ui(self):
        """禁用UI组件"""
        self.disable_widget_tree(self.main_frame)
        self.extract_button.config(state=tk.DISABLED)

    def enable_ui(self):
        """启用UI组件"""
        self.enable_widget_tree(self.main_frame)
        self.extract_button.config(state=tk.NORMAL)

    def disable_widget_tree(self, widget):
        """递归禁用小部件树"""
        try:
            widget.config(state=tk.DISABLED)
        except Exception:
            pass

        for child in widget.winfo_children():
            self.disable_widget_tree(child)

    def enable_widget_tree(self, widget):
        """递归启用小部件树"""
        try:
            widget.config(state=tk.NORMAL)
        except Exception:
            pass

        for child in widget.winfo_children():
            self.enable_widget_tree(child)


def main():
    """主函数"""
    # 检查是否可以使用TkinterDnD
    if TK_DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = VideoOCRUploadApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

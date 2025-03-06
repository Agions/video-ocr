#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频OCR字幕提取工具命令行界面
"""

import os
import sys
import argparse
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from main import VideoOCR


class VideoOCRApp:
    """视频OCR字幕提取工具图形界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("视频OCR字幕提取工具")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(
            main_frame, 
            text="视频OCR字幕提取工具", 
            font=("Arial", 18, "bold")
        ).pack(pady=10)
        
        # 文件选择框架
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        # 视频文件选择
        ttk.Label(file_frame, text="视频文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_video).grid(row=0, column=2, pady=5)
        
        # 输出文件选择
        ttk.Label(file_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, pady=5)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        # 字幕区域
        ttk.Label(param_frame, text="字幕区域:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.subtitle_area = tk.StringVar(value="0.7,1.0")
        ttk.Entry(param_frame, textvariable=self.subtitle_area, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(param_frame, text="格式: 开始高度,结束高度 (0.0-1.0)").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 识别语言
        ttk.Label(param_frame, text="识别语言:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lang = tk.StringVar(value="zh")
        lang_combo = ttk.Combobox(param_frame, textvariable=self.lang, width=8)
        lang_combo['values'] = ('zh', 'en', 'jp', 'korean')
        lang_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(param_frame, text="zh: 中文, en: 英文, jp: 日文, korean: 韩文").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 提取间隔
        ttk.Label(param_frame, text="提取间隔:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.interval = tk.StringVar(value="1.0")
        ttk.Entry(param_frame, textvariable=self.interval, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(param_frame, text="秒").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 最小持续时间
        ttk.Label(param_frame, text="最小持续时间:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.min_duration = tk.StringVar(value="0.3")
        ttk.Entry(param_frame, textvariable=self.min_duration, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(param_frame, text="秒").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 增强选项
        self.enhance_frames = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            param_frame, 
            text="增强帧以提高OCR识别率", 
            variable=self.enhance_frames
        ).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 调试选项
        self.debug = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            param_frame, 
            text="启用调试模式", 
            variable=self.debug
        ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 进度条
        progress_frame = ttk.Frame(main_frame, padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(progress_frame, text="处理进度:").pack(anchor=tk.W)
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=550, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # 状态信息
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            button_frame, 
            text="开始处理", 
            command=self.start_processing,
            style="Accent.TButton",
            width=20
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="退出", 
            command=self.root.destroy,
            width=10
        ).pack(side=tk.RIGHT, padx=5)
        
        # 创建按钮样式
        self.root.style = ttk.Style()
        self.root.style.configure("Accent.TButton", font=("Arial", 11, "bold"))
    
    def browse_video(self):
        """浏览选择视频文件"""
        filetypes = [
            ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
            ("所有文件", "*.*")
        ]
        video_file = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=filetypes
        )
        if video_file:
            self.video_path.set(video_file)
            # 自动设置输出文件路径
            video_dir = os.path.dirname(video_file)
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            output_path = os.path.join(video_dir, f"{video_name}.srt")
            self.output_path.set(output_path)
    
    def browse_output(self):
        """浏览选择输出文件"""
        filetypes = [
            ("字幕文件", "*.srt"),
            ("所有文件", "*.*")
        ]
        output_file = filedialog.asksaveasfilename(
            title="保存字幕文件",
            filetypes=filetypes,
            defaultextension=".srt"
        )
        if output_file:
            self.output_path.set(output_file)
    
    def start_processing(self):
        """开始处理视频"""
        # 检查输入
        if not self.video_path.get():
            messagebox.showerror("错误", "请选择视频文件")
            return
        
        if not self.output_path.get():
            messagebox.showerror("错误", "请选择输出文件路径")
            return
        
        # 检查视频文件是否存在
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("错误", f"视频文件不存在: {self.video_path.get()}")
            return
        
        # 解析参数
        try:
            subtitle_area = tuple(map(float, self.subtitle_area.get().split(',')))
            interval = float(self.interval.get())
            min_duration = float(self.min_duration.get())
            
            if len(subtitle_area) != 2:
                raise ValueError("字幕区域格式错误")
                
            if not (0 <= subtitle_area[0] < subtitle_area[1] <= 1):
                raise ValueError("字幕区域范围错误，应为0.0-1.0")
                
            if interval <= 0:
                raise ValueError("提取间隔必须大于0")
                
            if min_duration < 0:
                raise ValueError("最小持续时间不能小于0")
                
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        
        # 禁用界面
        for widget in self.root.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Button) or isinstance(child, ttk.Entry) or isinstance(child, ttk.Combobox) or isinstance(child, ttk.Checkbutton):
                    child.configure(state="disabled")
        
        # 启动进度条
        self.progress.start()
        self.status_var.set("正在处理视频，请稍候...")
        
        # 在后台线程中处理视频
        import threading
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
    
    def process_video(self):
        """在后台线程中处理视频"""
        try:
            # 创建输出目录
            output_dir = os.path.dirname(self.output_path.get())
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 初始化视频OCR工具
            ocr = VideoOCR(
                self.video_path.get(),
                subtitle_area=tuple(map(float, self.subtitle_area.get().split(','))),
                lang=self.lang.get(),
                interval=float(self.interval.get())
            )
            
            # 处理视频
            ocr.process_video(
                self.output_path.get(),
                min_duration=float(self.min_duration.get()),
                enhance_frames=self.enhance_frames.get(),
                debug=self.debug.get()
            )
            
            # 处理完成
            self.root.after(0, self.processing_complete, True, "处理完成！字幕已保存至：" + self.output_path.get())
            
        except Exception as e:
            import traceback
            error_msg = f"处理出错: {str(e)}\n{traceback.format_exc()}"
            self.root.after(0, self.processing_complete, False, error_msg)
    
    def processing_complete(self, success, message):
        """处理完成回调"""
        # 停止进度条
        self.progress.stop()
        
        # 启用界面
        for widget in self.root.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Button) or isinstance(child, ttk.Entry) or isinstance(child, ttk.Combobox) or isinstance(child, ttk.Checkbutton):
                    child.configure(state="normal")
        
        # 显示结果
        if success:
            self.status_var.set("处理完成")
            messagebox.showinfo("成功", message)
        else:
            self.status_var.set("处理出错")
            messagebox.showerror("错误", message)


def main_gui():
    """启动GUI界面"""
    root = tk.Tk()
    app = VideoOCRApp(root)
    root.mainloop()


def main_cli():
    """命令行界面"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='视频OCR字幕提取工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument('video', help='输入视频文件路径')
    parser.add_argument('-o', '--output', help='输出SRT文件路径，默认为与视频同名的SRT文件')
    parser.add_argument('-a', '--area', default='0.7,1.0',
                      help='字幕区域范围，格式为"开始高度比例,结束高度比例"')
    parser.add_argument('-l', '--lang', default='zh', choices=['zh', 'en', 'jp', 'korean'],
                      help='识别语言')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                      help='视频帧提取间隔（秒）')
    parser.add_argument('-d', '--min-duration', type=float, default=0.3,
                      help='最小字幕持续时间（秒）')
    parser.add_argument('--no-enhance', action='store_true',
                      help='不对帧进行增强处理')
    parser.add_argument('--debug', action='store_true',
                      help='输出调试信息')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        return 1
    
    # 如果未指定输出文件，则使用默认命名
    if not args.output:
        video_dir = os.path.dirname(args.video)
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(video_dir, f"{video_name}.srt")
    
    # 解析字幕区域参数
    subtitle_area = tuple(map(float, args.area.split(',')))
    
    try:
        # 创建输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"正在处理视频: {args.video}")
        print(f"输出字幕文件: {args.output}")
        
        # 计时开始
        start_time = time.time()
        
        # 初始化视频OCR工具
        ocr = VideoOCR(
            args.video,
            subtitle_area=subtitle_area,
            lang=args.lang,
            interval=args.interval
        )
        
        # 处理视频
        ocr.process_video(
            args.output,
            min_duration=args.min_duration,
            enhance_frames=not args.no_enhance,
            debug=args.debug
        )
        
        # 计时结束
        end_time = time.time()
        print(f"总耗时: {end_time - start_time:.2f}秒")
        
    except KeyboardInterrupt:
        print("\n处理被用户中断")
        return 130
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # 判断是否由py2app启动
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 作为应用程序运行时启动GUI
        main_gui()
    elif len(sys.argv) > 1:
        # 有命令行参数时使用CLI模式
        sys.exit(main_cli())
    else:
        # 没有参数时启动GUI
        main_gui() 
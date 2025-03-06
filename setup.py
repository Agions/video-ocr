from setuptools import setup

APP = ['video_upload_interface.py']
DATA_FILES = [
    'main.py',
    'video_utils.py',
    'ocr_utils.py',
    'subtitle_utils.py',
    'extract_subtitle.py',
    'example.py',
    'README.md'
]
OPTIONS = {
    'argv_emulation': True,
    'packages': ['paddleocr', 'paddlepaddle', 'cv2', 'numpy', 'pysrt', 'skimage', 'tqdm', 'tkinter', 'PIL'],
    'includes': ['tkinter', 'PIL._tkinter_finder'],
    'iconfile': 'app_icon.icns',
    'plist': {
        'CFBundleName': '视频OCR字幕提取工具',
        'CFBundleDisplayName': '视频OCR字幕提取工具',
        'CFBundleGetInfoString': "视频OCR字幕提取工具",
        'CFBundleIdentifier': "com.videoocr.subtitleextractor",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0.0",
        'NSHumanReadableCopyright': "Copyright © 2023, 视频OCR字幕提取工具",
    },
    'resources': [],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

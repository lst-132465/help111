import whisper
import os
import torch
from typing import Optional
import streamlit as st  # 【新增】导入Streamlit，用于缓存和加载提示

# 全局变量，避免重复加载模型（优化性能）
WHISPER_MODEL = None
SUPPORTED_AUDIO_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg"}

def init_whisper(model_name: str = "base", device: Optional[str] = None) -> None:
    """
    初始化 Whisper 模型（适配 openai-whisper==20230314）
    首次调用时加载，后续复用全局模型实例
    """
    global WHISPER_MODEL
    
    # 检查是否已加载模型
    if WHISPER_MODEL is not None:
        return
    
    try:
        # 自动选择设备（CPU，适配 Streamlit Cloud）
        if device is None:
            device = "cpu"
        
        # 【新增】使用 Streamlit 缓存资源，避免重复加载（关键优化）
        @st.cache_resource(show_spinner=False)
        def _load_whisper_model(_model_name: str, _device: str):
            """内部缓存函数，仅加载一次模型"""
            return whisper.load_model(
                name=_model_name,
                device=_device,
                # 【优化】使用系统临时目录，适配 Streamlit Cloud 存储限制
                download_root=os.path.join(os.getenv("TMPDIR", os.getcwd()), "whisper_models")
            )
        
        # 【新增】显示加载提示，提升用户体验
        with st.spinner("正在加载语音识别模型（首次使用需1-2分钟）..."):
            WHISPER_MODEL = _load_whisper_model(model_name, device)
            print(f"✅ Whisper 模型 '{model_name}' 已成功加载到 {device}")
        
    except Exception as e:
        error_msg = f"❌ Whisper 模型初始化失败: {str(e)}"
        print(error_msg)
        # 抛出异常供上层处理
        raise RuntimeError(error_msg) from e

def audio_to_text(audio_path: str, language: str = "zh", task: str = "transcribe") -> str:
    """
    音频转文字（适配 openai-whisper==20230314）
    Args:
        audio_path: 音频文件路径
        language: 语言代码（zh=中文，en=英文）
        task: 任务类型（transcribe=转录，translate=翻译）
    Returns:
        转录文本
    """
    global WHISPER_MODEL
    
    # 前置检查
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    file_ext = audio_path.split(".")[-1].lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"不支持的音频格式: {file_ext}，支持格式: {SUPPORTED_AUDIO_FORMATS}")
    
    # 确保模型已初始化
    if WHISPER_MODEL is None:
        init_whisper()
    
    try:
        # 【新增】显示转写提示，避免用户误以为页面无响应
        with st.spinner("正在解析音频内容..."):
            # 关键适配：20230314 版本的 transcribe 参数
            result = WHISPER_MODEL.transcribe(
                audio=audio_path,
                language=language,
                task=task,
                verbose=False,  # 禁用详细输出，避免 Streamlit 日志刷屏
                word_timestamps=False,  # 禁用词级时间戳，提升速度
                fp16=False  # CPU 环境必须禁用 fp16
            )
        
        # 提取纯文本结果
        full_text = result.get("text", "").strip()
        
        # 拼接段落（如果有）
        if "segments" in result:
            segment_texts = [seg.get("text", "").strip() for seg in result["segments"]]
            full_text = " ".join(segment_texts)
        
        return full_text if full_text else "⚠️ 未检测到有效语音内容"
        
    except Exception as e:
        error_msg = f"❌ 音频转文字失败: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def is_whisper_available() -> bool:
    """检查 Whisper 是否可用"""
    try:
        import whisper
        return True
    except ImportError:
        return False

# 【关键修改】删除启动时预加载模型的代码，改为用户首次调用时加载
# （原有预加载代码已删除，不影响功能，仅避免启动超时）

import whisper
import os
import torch
from typing import Optional
import streamlit as st

# 全局变量，避免重复加载模型（优化性能）
WHISPER_MODEL = None
SUPPORTED_AUDIO_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg"}

# 用 Streamlit 缓存资源装饰器，实现「延迟加载+全局缓存」
@st.cache_resource
def _load_whisper_model(model_name: str = "base", device: Optional[str] = None):
    """内部函数：加载并缓存模型，仅首次调用时执行"""
    if device is None:
        device = "cpu"
    
    # 用 Streamlit 临时目录存储模型，避免权限问题
    model_cache_dir = os.path.join(os.getenv("TMPDIR", os.getcwd()), "whisper_models")
    os.makedirs(model_cache_dir, exist_ok=True)
    
    model = whisper.load_model(
        name=model_name,
        device=device,
        download_root=model_cache_dir
    ).float()  # 强制转为float32，适配CPU环境
    
    print(f"✅ Whisper 模型 '{model_name}' 已成功加载到 {device} (float32模式)")
    return model

def init_whisper(model_name: str = "base", device: Optional[str] = None) -> None:
    """兼容原有接口，不修改app.py中的调用方式"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = _load_whisper_model(model_name, device)

def audio_to_text(audio_path: str, language: str = "zh", task: str = "transcribe") -> str:
    """
    音频转文字（接口完全不变，适配openai-whisper==20230314）
    """
    global WHISPER_MODEL
    
    # 前置检查
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    file_ext = audio_path.split(".")[-1].lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"不支持的音频格式: {file_ext}，支持格式: {SUPPORTED_AUDIO_FORMATS}")
    
    # 确保模型已初始化（首次调用时才加载，避免启动时OOM）
    if WHISPER_MODEL is None:
        init_whisper()
    
    try:
        result = WHISPER_MODEL.transcribe(
            audio=audio_path,
            language=language,
            task=task,
            verbose=False,
            word_timestamps=False,
            fp16=False,  # CPU 环境必须禁用 fp16
            temperature=0.0,
            best_of=1  # 减少内存占用，提升稳定性
        )
        
        full_text = result.get("text", "").strip()
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

# 移除启动时预加载，改为用户首次调用时加载，避免启动超时
# （原有预加载代码已删除，不影响功能，仅优化部署稳定性）

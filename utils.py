import whisper
import os
import torch
from typing import Optional
import ffmpeg  # 新增：显式导入ffmpeg-python，确保依赖检测通过

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
        
        # 关键适配1：Streamlit Cloud 路径优化，使用临时目录存储模型（避免权限问题）
        model_cache_dir = os.path.join(os.getenv("TMPDIR", os.getcwd()), "whisper_models")
        os.makedirs(model_cache_dir, exist_ok=True)  # 确保目录存在
        
        # 关键适配2：20230314 版本使用 load_model API + CPU 强制 float32
        WHISPER_MODEL = whisper.load_model(
            name=model_name,
            device=device,
            download_root=model_cache_dir
        ).float()  # 强制转为float32，避免CPU环境FP16警告与错误
        
        print(f"✅ Whisper 模型 '{model_name}' 已成功加载到 {device} (float32模式)")
        
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
        # 关键适配3：20230314 版本参数锁定，CPU 禁用 fp16，添加温度控制提升稳定性
        result = WHISPER_MODEL.transcribe(
            audio=audio_path,
            language=language,
            task=task,
            verbose=False,  # 禁用详细输出，避免 Streamlit 日志刷屏
            word_timestamps=False,  # 禁用词级时间戳，提升速度
            fp16=False,  # CPU 环境必须禁用 fp16（20230314 版本关键参数）
            temperature=0.0,  # 固定温度，避免随机波动，提升部署稳定性
            best_of=1  # 减少候选数，降低内存占用
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
        import ffmpeg  # 新增：检查ffmpeg依赖
        return True
    except ImportError:
        return False

# 可选：预加载模型（应用启动时执行）
if is_whisper_available():
    try:
        init_whisper()
    except Exception as e:
        print(f"⚠️ 预加载 Whisper 模型失败（非致命错误）: {str(e)}")

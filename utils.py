from faster_whisper import WhisperModel
import os
from typing import Optional

# 全局变量，避免重复加载模型（与原逻辑完全一致）
WHISPER_MODEL = None
SUPPORTED_AUDIO_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg"}

def init_whisper(model_name: str = "base", device: Optional[str] = None) -> None:
    """与原接口完全一致，仅替换底层实现"""
    global WHISPER_MODEL
    if WHISPER_MODEL is not None:
        return
    
    if device is None:
        device = "cpu"
    
    # 自定义模型缓存路径，避免权限问题
    model_cache_dir = os.path.join(os.getenv("TMPDIR", os.getcwd()), "whisper_models")
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # faster-whisper 模型加载（与原接口参数兼容）
    WHISPER_MODEL = WhisperModel(
        model_name,
        device=device,
        compute_type="int8",  # CPU 环境最优量化方式
        download_root=model_cache_dir
    )
    print(f"✅ Faster-Whisper 模型 '{model_name}' 已成功加载到 {device}")

def audio_to_text(audio_path: str, language: str = "zh", task: str = "transcribe") -> str:
    """与原接口完全一致，不改变任何调用方式"""
    global WHISPER_MODEL
    
    # 前置检查（与原逻辑完全一致）
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    file_ext = audio_path.split(".")[-1].lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"不支持的音频格式: {file_ext}，支持格式: {SUPPORTED_AUDIO_FORMATS}")
    
    # 确保模型已初始化（与原逻辑完全一致）
    if WHISPER_MODEL is None:
        init_whisper()
    
    try:
        # faster-whisper 转录（参数与原逻辑兼容，返回格式一致）
        segments, info = WHISPER_MODEL.transcribe(
            audio_path,
            language=language,
            vad_filter=True  # 过滤静音，提升识别准确率
        )
        
        # 拼接文本，与原函数返回格式完全一致
        full_text = " ".join([segment.text.strip() for segment in segments])
        return full_text if full_text else "⚠️ 未检测到有效语音内容"
        
    except Exception as e:
        error_msg = f"❌ 音频转文字失败: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def is_whisper_available() -> bool:
    """检查 Whisper 是否可用（与原逻辑完全一致）"""
    try:
        from faster_whisper import WhisperModel
        return True
    except ImportError:
        return False

# 移除启动时预加载，改为用户首次调用时加载，避免启动超时

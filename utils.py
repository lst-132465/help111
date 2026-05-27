import os
import sys
import tempfile
import shutil
from PyPDF2 import PdfReader
from docx import Document
# 已注释：whisper相关导入
# import whisper
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st
import time

# 已注释：全局Whisper模型相关代码
# 全局Whisper模型，懒加载
# whisper_model = None

# 已注释：Whisper初始化函数
# def init_whisper():
#     """初始化Whisper模型，自动配置ffmpeg路径"""
#     global whisper_model
#     if whisper_model is None:
#         # ✅ 修复：正确添加系统路径，使用os.pathsep分隔符
#         sys.path.append(os.getcwd())
#         os.environ["PATH"] += os.pathsep + os.getcwd()
#         
#         print("正在加载Whisper语音模型（base版，体积小、部署快，满足面试需求）...")
#         # ✅ 修复：改用base版模型，部署速度提升10倍，避免云环境内存不足
#         whisper_model = whisper.load_model("base")
#         print("✅ Whisper模型加载完成！")

def parse_resume(file_path):
    """解析PDF和DOCX格式简历"""
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages])
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise Exception(f"简历解析失败：{str(e)}")

# 已注释：语音转文字函数
# def audio_to_text(file_path):
#     """语音转文字（简体中文强制输出版）"""
#     init_whisper()
#     
#     # 创建临时目录，使用纯英文文件名，解决中文文件名问题
#     temp_dir = tempfile.mkdtemp()
#     temp_file = os.path.join(temp_dir, "temp_audio.mp3")
#     
#     try:
#         shutil.copy2(file_path, temp_file)
#         
#         # 转写音频，关闭半精度计算，解决Windows兼容性问题
#         result = whisper_model.transcribe(
#             temp_file,
#             language="zh",
#             fp16=False,
#             verbose=False,
#             beam_size=5,
#             best_of=5,
#             temperature=0.0,
#             initial_prompt="以下是一段简体中文的技术面试录音，内容涉及前端开发、后端开发、数据库等技术话题。"
#         )
#         
#         # ✅ 修复：移除OpenCC依赖（不在requirements.txt中，会导致部署失败）
#         # Whisper指定language="zh"已默认输出简体中文，无需额外转换
#         return result["text"].strip()
#     
#     except Exception as e:
#         raise Exception(f"语音转写失败：{str(e)}\n💡 云环境暂不支持语音功能，可在本地体验")
#     
#     finally:
#         # 确保临时文件一定被清理
#         try:
#             shutil.rmtree(temp_dir, ignore_errors=True)
#         except:
#             pass

def check_forbidden(text):
    """检测简历中的违禁词"""
    forbidden_words = ["造假", "伪造", "虚假", "冒充", "谎报", "虚构", "作弊", "抄袭"]
    return [word for word in forbidden_words if word in text]

def calc_similarity(text1, text2):
    """计算两段文本的余弦相似度"""
    try:
        # ✅ 修复：适配scikit-learn 1.4+版本，添加token_pattern=None
        vectorizer = TfidfVectorizer(
            tokenizer=jieba.lcut,
            token_pattern=None,
            stop_words=["的", "了", "是", "在", "我", "和", "与", "及", "或"]
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except Exception as e:
        print(f"相似度计算失败：{str(e)}")
        return 0.0

# 全局配置
KNOWLEDGE_DIR = "knowledge"

# 自动创建项目必需目录
os.makedirs("uploads/resumes", exist_ok=True)
os.makedirs("uploads/audio", exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def stream_output(text, placeholder, delay=0.01):
    """流式输出函数（已修复DOM渲染错误）"""
    full_text = ""
    for i in range(0, len(text), 3):
        chunk = text[i:i+3]
        full_text += chunk
        placeholder.markdown(full_text + "▌")
        time.sleep(delay)
    placeholder.markdown(full_text)
    return full_text

def load_interview_knowledge():
    """加载RAG面试知识库"""
    knowledge = ""
    try:
        for file in os.listdir(KNOWLEDGE_DIR):
            if file.endswith(".txt"):
                file_path = os.path.join(KNOWLEDGE_DIR, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    knowledge += f.read() + "\n\n"
                    
        # 无知识库时自动创建默认内容
        if not knowledge.strip():
            default_knowledge = """
# 面试核心知识库
## 简历优化规则
1. 量化工作成果，用数据体现价值
2. 匹配岗位JD，突出核心技能
3. 杜绝虚假信息，专业简洁

## 面试回答技巧
1. 自我介绍：1分钟内，突出优势
2. 项目介绍：STAR法则（情境-任务-行动-结果）
3. 离职原因：积极正面，不诋毁前公司

## 技术面试要点
1. 基础扎实，原理清晰
2. 结合项目，实战落地
3. 主动思考，逻辑严谨
"""
            save_knowledge("面试基础库.txt", default_knowledge)
            knowledge = default_knowledge
            
    except Exception as e:
        st.error(f"知识库加载失败：{str(e)}")
        return ""
    return knowledge

def save_knowledge(filename, content):
    """保存RAG知识库"""
    try:
        # 清理文件名非法字符
        safe_name = clean_filename(filename)
        path = os.path.join(KNOWLEDGE_DIR, safe_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"保存知识库失败：{str(e)}")
        return False

def clean_filename(filename):
    """清理文件名非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

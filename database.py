import sqlite3
import os

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interview.db")  # 【修复1】使用绝对路径，避免Streamlit环境下路径错误

def get_conn():
    """获取数据库连接
    说明：check_same_thread=False 是Streamlit多线程环境下的标准配置
    所有数据库操作都通过本模块函数封装，保证线程安全
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # 支持字典格式读取数据
    return conn

def init_database():
    """初始化数据库：所有表结构100%保留，仅修复潜在问题"""
    conn = get_conn()
    c = conn.cursor()

    # ========== 原有所有表 完全保留 不做任何修改 ==========
    # 简历评估表
    c.execute('''CREATE TABLE IF NOT EXISTS resumes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT, content TEXT, similarity REAL,
                  forbidden_words INTEGER, score REAL, report TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 面试录音分析表
    c.execute('''CREATE TABLE IF NOT EXISTS interviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT, transcript TEXT, score REAL, report TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 通用题库表
    c.execute('''CREATE TABLE IF NOT EXISTS questions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT, answer TEXT, source TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 个性化面试题生成历史表
    c.execute('''CREATE TABLE IF NOT EXISTS generated_questions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_filename TEXT,
                  questions TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 实时面试模拟历史表
    c.execute('''CREATE TABLE IF NOT EXISTS mock_interviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_filename TEXT,
                  final_score REAL,
                  full_report TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 求职信生成历史表
    c.execute('''CREATE TABLE IF NOT EXISTS cover_letters
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job TEXT,
                  company TEXT,
                  resume_filename TEXT,
                  content TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 薪资谈判历史表
    c.execute('''CREATE TABLE IF NOT EXISTS salary_advice
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job TEXT,
                  city TEXT,
                  experience INTEGER,
                  content TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 职业规划历史表
    c.execute('''CREATE TABLE IF NOT EXISTS career_plans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  interest TEXT,
                  resume_filename TEXT,
                  content TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # ========== 扩展功能数据表 完全保留 ==========
    # 简历ATS优化记录表
    c.execute('''CREATE TABLE IF NOT EXISTS ats_optimizations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_filename TEXT,
                  target_jd TEXT,
                  ats_score REAL,
                  optimization_report TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 多风格面试模拟记录表
    c.execute('''CREATE TABLE IF NOT EXISTS multi_style_interviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_filename TEXT,
                  interview_style TEXT,
                  conversation_history TEXT,
                  final_score REAL,
                  review_report TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 统一智能助手对话历史表
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_message TEXT,
                  assistant_response TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # ========== 国赛新增表 完全保留 ==========
    # 中央调度任务表
    c.execute('''CREATE TABLE IF NOT EXISTS scheduler_tasks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_filename TEXT,
                  task_log TEXT,
                  task_status TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # RAG知识库管理表
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_base
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  knowledge_type TEXT,
                  content TEXT,
                  update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

# ========== 数据统计函数（修复边界情况） ==========
def get_dashboard_stats():
    """获取全局数据统计，用于前端数据看板"""
    conn = get_conn()
    stats = {}
    
    # 所有统计都添加COALESCE处理空表情况
    stats["resume_count"] = conn.execute("SELECT COALESCE(COUNT(*), 0) FROM resumes").fetchone()[0]
    stats["mock_count"] = conn.execute("SELECT COALESCE(COUNT(*), 0) FROM mock_interviews").fetchone()[0]
    stats["audio_count"] = conn.execute("SELECT COALESCE(COUNT(*), 0) FROM interviews").fetchone()[0]
    stats["ats_count"] = conn.execute("SELECT COALESCE(COUNT(*), 0) FROM ats_optimizations").fetchone()[0]
    
    avg_score = conn.execute("SELECT AVG(score) FROM resumes").fetchone()[0]
    stats["avg_resume_score"] = round(avg_score, 1) if avg_score else 0
    
    conn.close()
    return stats

# ========== 通用历史记录查询函数（修复SQL注入风险） ==========
def get_history(table_name, limit=10):
    """通用查询历史记录（添加表名白名单，防止SQL注入）"""
    # 允许查询的表白名单
    ALLOWED_TABLES = {
        "resumes", "interviews", "generated_questions", "mock_interviews",
        "cover_letters", "salary_advice", "career_plans", "ats_optimizations",
        "multi_style_interviews", "chat_history", "scheduler_tasks", "knowledge_base"
    }
    
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"不允许查询表：{table_name}")
    
    conn = get_conn()
    cursor = conn.cursor()
    # 使用参数化查询传递limit参数
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

# ========== RAG知识库操作函数（完全保留） ==========
def save_knowledge(knowledge_type, content):
    """保存知识库内容（追加模式，保留历史版本）"""
    conn = get_conn()
    conn.execute(
        "INSERT INTO knowledge_base (knowledge_type, content) VALUES (?, ?)",
        (knowledge_type, content)
    )
    conn.commit()
    conn.close()

def get_all_knowledge():
    """获取所有知识库内容"""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM knowledge_base ORDER BY update_time DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

# ========== 调度任务保存函数（完全保留） ==========
def save_scheduler_task(resume_filename, task_log, task_status="完成"):
    """保存调度Agent执行日志"""
    conn = get_conn()
    conn.execute(
        "INSERT INTO scheduler_tasks (resume_filename, task_log, task_status) VALUES (?, ?, ?)",
        (resume_filename, str(task_log), task_status)
    )
    conn.commit()
    conn.close()

# 初始化数据库（幂等操作，多次执行安全）
init_database()

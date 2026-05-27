环境要求

Python 3.9 \~ 3.11（推荐 3.11，兼容性最好），至少 4GB 内存

第一步：打开 CMD 并进入项目文件夹

按 Win + R 键，输入 cmd 并回车，打开命令提示符窗口

根据你的项目所在盘符执行对应命令：

如果项目在 C 盘（如你的桌面）：

cmd

cd C:\\Users\\LST33\\Desktop\\智能面试助手文件

如果项目在 其他盘（如 D 盘）：

cmd

D:

cd D:\\你的项目文件夹路径

执行成功后，CMD 的路径会变成：

plaintext

C:\\Users\\LST33\\Desktop\\智能面试助手文件>

第二步：安装所有依赖

在上面的 CMD 窗口中，复制粘贴以下完整命令并回车：

cmd

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

等待 1-2 分钟，看到 Successfully installed 字样即表示安装完成。

第三步：配置环境变量

回到项目文件夹，复制 .env.example 文件

将复制后的文件重命名为 .env  

用记事本打开 .env 文件，填入你的大模型 API 密钥（至少配置智谱清言即可运行所有功能）：

ini

\# 智谱清言（必配，系统默认使用）

ZHIPU\_API\_KEY=你的智谱清言API密钥



\# 以下为可选配置

XF\_API\_KEY=你的讯飞星火API密钥

QWEN\_API\_KEY=你的通义千问API密钥

DOUBAO\_API\_KEY=你的字节跳动豆包API密钥

保存并关闭 .env 文件



注：第三步骤传输的API密钥文件已完成配置



第四步：启动应用

回到刚才的 CMD 窗口，执行以下命令启动系统：

cmd

streamlit run app.py

访问地址

启动成功后，浏览器会自动打开以下地址，即可使用所有功能：

plaintext

http://localhost:8501

常见问题解决

提示 "pip 不是内部或外部命令"

解决方法：使用 python -m pip 代替 pip，安装命令变为：

cmd

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

提示 "找不到 openai\_api\_key"

解决方法：确认你已经将 .env.example 重命名为 .env，并且在文件中正确填写了 ZHIPU\_API\_KEY。

端口 8501 被占用

解决方法：指定其他端口启动：

cmd

streamlit run app.py --server.port 8502


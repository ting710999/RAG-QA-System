import os
import shutil
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi

#  API 配置
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print(f"数据文件夹: {DATA_FOLDER}")
print(f"文件夹存在: {os.path.exists(DATA_FOLDER)}")

if os.path.exists(DATA_FOLDER):
    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    print(f"TXT文件列表: {txt_files}")

#  Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5"
)

# 全局缓存
_docs = None
_bm25 = None
_vector_db = None


#  1. 加载文档
def load_docs(force_reload=False):
    global _docs
    if _docs is not None and not force_reload:
        return _docs

    print("加载文档...")

    # 尝试不同编码
    encodings = ['utf-8', 'gbk', 'gb2312']
    docs = None

    for encoding in encodings:
        try:
            loader = DirectoryLoader(
                DATA_FOLDER,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": encoding}
            )
            docs = loader.load()
            if len(docs) > 0:
                print(f"使用编码 {encoding} 成功加载 {len(docs)} 个文档")
                break
        except Exception as e:
            print(f"编码 {encoding} 失败: {e}")
            continue

    if not docs or len(docs) == 0:
        raise Exception(f"未找到任何 txt 文件，请检查 {DATA_FOLDER} 文件夹")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    _docs = splitter.split_documents(docs)
    print(f"切分完成，共 {len(_docs)} 个文本块")

    # 打印第一个文档预览
    if len(_docs) > 0:
        print(f"文档预览: {_docs[0].page_content[:100]}...")

    return _docs


#  2. 向量库
def load_db():
    global _vector_db
    if _vector_db is not None:
        return _vector_db

    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        print("加载已有向量库...")
        _vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        return _vector_db

    print("创建向量库...")
    docs = load_docs()
    _vector_db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    _vector_db.persist()
    print("向量库创建成功")
    return _vector_db


#  3. BM25 索引
def init_bm25():
    global _bm25
    if _bm25 is not None:
        return _bm25

    docs = load_docs()
    corpus = [d.page_content for d in docs]
    tokenized = [doc.split() for doc in corpus]
    _bm25 = BM25Okapi(tokenized)
    print("BM25 索引创建成功")
    return _bm25


#  4. LLM 调用
def llm_answer(question, context=""):
    prompt = f"""你是一个严格基于资料回答的RAG系统。

【资料】
{context}

【问题】
{question}

规则：
- 只能使用资料内容
- 资料没有 → 说"知识库未找到相关信息"
- 不允许编造
"""

    try:
        res = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"LLM调用失败：{str(e)}"


# 5. RAG 核心
def run_agent(question):
    print(f"\nRAG启动: {question}")

    db = load_db()
    bm25 = init_bm25()
    docs = load_docs()

    # 向量检索
    vector_results = db.similarity_search_with_score(question, k=3)
    best_doc, score = vector_results[0]
    vector_score = 1 / (1 + score)

    # BM25检索
    bm25_scores = bm25.get_scores(question.split())
    best_bm25_idx = bm25_scores.argmax()
    bm25_score = bm25_scores[best_bm25_idx]

    print(f"向量分数: {vector_score:.4f}")
    print(f"BM25分数: {bm25_score:.4f}")

    context = best_doc.page_content

    if vector_score > 0.55 or bm25_score > 5:
        print("有证据 → RAG回答")
        return llm_answer(question, context)
    elif vector_score > 0.35:
        print("弱证据 → RAG+AI")
        return llm_answer(question, context)
    else:
        print("无证据 → 拒绝幻觉")
        return "知识库中未找到相关信息，请补充资料后再查询。"


#  6. 对外接口
def ask(question):
    return run_agent(question)


def ask_with_status(question):
    result = run_agent(question)
    return {"found": True, "answer": result, "message": "查询完成"}


# 7. 重建向量库
def rebuild_db():
    global _vector_db, _docs, _bm25

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("旧向量库已删除")

    _vector_db = None
    _docs = None
    _bm25 = None

    load_db()
    print("向量库重建完成")


#  8. 测试
if __name__ == "__main__":
    # 重建向量库
    rebuild_db()

    print("\n" + "=" * 60)
    print("RAG 系统测试")
    print("=" * 60)

    test_questions = [
        "什么是RAG？",
        "什么是Agent？",
        "Flask是什么？"
    ]

    for q in test_questions:
        print(f"\n问题: {q}")
        answer = ask(q)
        print(f"答案: {answer}\n")
        print("-" * 60)
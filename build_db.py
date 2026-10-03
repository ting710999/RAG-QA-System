import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader

# 禁用chroma遥测，消除报错
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# 配置
DB_PATH = "./vector_db"
DATA_PATH = "./data"

# 嵌入模型（统一全局配置）
def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# 重建向量库
def build_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("已删除旧向量库")

    # 加载所有txt
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"加载文件：{filename}")

    # 分块（统一最优参数）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    splits = splitter.split_documents(documents)
    print(f"文本分块完成：{len(splits)} 块")

    # 入库
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=DB_PATH
    )
    vector_db.persist()
    print("向量库构建完成！")

if __name__ == "__main__":
    build_db()
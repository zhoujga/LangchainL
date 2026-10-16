import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"


def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)


def build_rag_chain(retriever, model):
    prompt = ChatPromptTemplate.from_template(
        "只能根据以下资料回答问题，不要编造。若资料没有答案，请明确说明。\n\n"
        "资料：\n{context}\n\n问题：\n{question}"
    )
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def load_retriever():
    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f"找不到知识目录：{DOCS_DIR}")

    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=False,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"知识目录中没有 Markdown 文件：{DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    splits = splitter.split_documents(documents)
    if not splits:
        raise ValueError("知识文件为空，无法建立向量索引。")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=None,
        collection_name="refund-rag-demo",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def main():
    parser = argparse.ArgumentParser(description="本地 Chroma + DeepSeek RAG 示例")
    parser.add_argument("--question", help="要询问的问题")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只测试本地向量检索，不调用聊天模型",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    question = args.question or input("请输入问题：").strip()
    if not question:
        raise ValueError("问题不能为空。")

    retriever = load_retriever()
    results = retriever.invoke(question)
    print(f"\n检索到 {len(results)} 个片段：")
    for index, document in enumerate(results, start=1):
        print(f"\n--- 片段 {index} ({document.metadata.get('source', 'unknown')}) ---")
        print(document.page_content)

    if args.retrieval_only:
        return
    if not results:
        print("没有检索到相关资料，已跳过模型调用。")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请在项目 .env 中设置 DEEPSEEK_API_KEY。")

    model = init_chat_model(
        "deepseek-flash",
        model_provider="openai",
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    rag_chain = build_rag_chain(retriever, model)
    answer = rag_chain.invoke(question)
    print("\n回答：")
    print(answer)


if __name__ == "__main__":
    main()

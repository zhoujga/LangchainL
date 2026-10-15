import math
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = PROJECT_DIR / "knowledge.txt"
load_dotenv(PROJECT_DIR / ".env")


def tokenize(text):
    tokens = []
    for segment in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(segment[index:index + 2] for index in range(len(segment) - 1))
            if len(segment) == 1:
                tokens.append(segment)
        else:
            tokens.append(segment)
    return tokens


def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start = end - overlap
    return chunks


def retrieve(question, chunks, top_k=3):
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    if not tokenized_chunks:
        return []

    document_frequency = Counter(
        token for tokens in tokenized_chunks for token in set(tokens)
    )
    average_length = sum(map(len, tokenized_chunks)) / len(tokenized_chunks) or 1
    query_tokens = set(tokenize(question))
    scored_chunks = []

    for chunk, tokens in zip(chunks, tokenized_chunks):
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            doc_frequency = document_frequency[token]
            inverse_frequency = math.log(
                1 + (len(chunks) - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            length_factor = 1.5 * (1 - 0.75 + 0.75 * len(tokens) / average_length)
            score += inverse_frequency * frequency * 2.5 / (frequency + length_factor)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]


def main():
    api_key = os.getenv("OPENAI_API_KEY1")
    if not api_key:
        raise RuntimeError("请先在项目 .env 文件中设置 OPENAI_API_KEY。")
    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f"未找到知识库文件：{KNOWLEDGE_FILE}")

    question = input("请输入问题：").strip()
    if not question:
        raise ValueError("问题不能为空。")

    knowledge = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    chunks = split_text(knowledge)
    retrieved_chunks = retrieve(question, chunks)
    context = "\n\n".join(retrieved_chunks) or "没有检索到相关知识。"

    model = init_chat_model(
        "openai:agnes-2.5-flash",
        api_key=api_key,
        base_url="https://apihub.agnes-ai.com/v1",
    )
    prompt = f"""请根据提供的知识库内容回答问题。如果知识库没有相关信息，请明确说明。

知识库内容：
{context}

问题：
{question}"""
    response = model.invoke(prompt)
    print("\n检索到的知识：")
    print(context)
    print("\n回答：")
    print(response.content)


if __name__ == "__main__":
    main()

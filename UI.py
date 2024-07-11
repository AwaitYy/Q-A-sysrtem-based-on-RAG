import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import random
from http import HTTPStatus
from dashscope import Generation
import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# 从多个PDF文件中提取文本
def read_pdfs(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        print("读取", pdf_path)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            text += doc.page_content
    print("完成资料读取")
    return text

# 对文档进行预处理，分割成多个小的文本块
def preprocess_document(document_text, split_docs_path):
    base_path = "documents"
    loaders = [PyPDFLoader(os.path.join(base_path, filename)) for filename in os.listdir(base_path)]
    documents = [loader.load() for loader in loaders]
    documents = [item for sublist in documents for item in sublist]  # 平展列表
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    text_chunks = text_splitter.split_text(document_text)
    with open(split_docs_path, 'wb') as f:
        pickle.dump(text_chunks, f)
    print("文档分割并保存成功")
    return text_chunks

# 使用指定的模型生成文本块的嵌入
def get_embeddings(text_chunks, model_name='bge-small-zh-v1.5'):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    return embeddings

# 将嵌入向量存储到Faiss索引中
def store_embeddings_in_faiss(embeddings, index_path="data/document_index.faiss"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = embeddings.cpu().detach().numpy()
    index.add(embeddings_np)
    faiss.write_index(index, index_path)
    print("向量库保存成功")
    return index

# 加载Faiss索引
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index

# 编码查询，生成其嵌入向量
def encode_query(query, model_name='bge-small-zh-v1.5'):
    embedder = SentenceTransformer(model_name)
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    return query_embedding

# 在Faiss索引中检索相似的嵌入向量
def search_similar_embeddings(query_embedding, index, top_k=3):
    query_embedding_np = query_embedding.cpu().detach().numpy()
    distances, indices = index.search(query_embedding_np, top_k)
    return indices

# 调用大模型生成答案
def call_with_aircraft_material_question(question, document_paragraphs):
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # API Key
    messages = [
        {'role': 'system', 'content': '你是一个高级机务工程师，专门回答有关飞机材料的问题。能够理解和分析技术文档中的复杂信息，并提供准确的答案。根据所提供的问题和相关的文档段落，生成一个简洁、准确的答案。'},
        {'role': 'user', 'content': f'回答该问题: {question}'}
    ]
    for paragraph in document_paragraphs:
        messages.append({'role': 'user', 'content': f'这是参考的文本：{paragraph}'})
    response = Generation.call(
        model="qwen-max",
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='message'
    )
    if response.status_code == HTTPStatus.OK:
        return response.output['choices'][0]['message']['content']
    else:
        return 'Request failed. Status code: %s, error message: %s' % (
            response.status_code, response.message
        )

# 处理单次提问-回复操作
def handle_query(query, text_chunks, index):
    query_embedding = encode_query(query)
    indices = search_similar_embeddings(query_embedding, index)
    similar_texts = [text_chunks[i] for i in indices[0]]
    response = call_with_aircraft_material_question(query, similar_texts)
    return similar_texts, response

# 主函数，用于初始化数据并处理用户查询
def main():
    pdf_paths = [
        "documents/M1-航空概论R1.pdf",
        "documents/M2-航空器维修R1.pdf",
        "documents/M3-飞机结构和系统R1.pdf",
        "documents/M4-直升机结构和系统.pdf",
        "documents/M5-航空涡轮发动机R1.pdf",
        "documents/M6-活塞发动机及其维修.pdf",
        "documents/M7-航空器维修基本技能.pdf",
        "documents/M8-航空器维修实践R1.pdf"
    ]
    index_path = "data/document_index.faiss"
    split_docs_path = "data/split_docs.pkl"
    if os.path.exists(split_docs_path):
        with open(split_docs_path, 'rb') as f:
            text_chunks = pickle.load(f)
        print("加载分割后的文档")
    else:
        print("读取文档并分割保存")
        document_text = read_pdfs(pdf_paths)
        text_chunks = preprocess_document(document_text, split_docs_path)

    if os.path.exists(index_path):
        print("加载向量数据库")
        index = load_faiss_index(index_path)
    else:
        print("生成嵌入并存储到向量数据库")
        embeddings = get_embeddings(text_chunks)
        index = store_embeddings_in_faiss(embeddings, index_path)

    return text_chunks, index

# 加载数据
text_chunks, index = main()

# 使用Gradio创建UI界面
import gradio as gr

def echo(message, history):
    similar_texts, response = handle_query(message, text_chunks, index)
    formatted_texts = "\n\n".join([f"Text chunk {i+1}:\n{text}" for i, text in enumerate(similar_texts)])
    return f"{response}"

examples = ["刚性主旋翼桨毂的优点", "AS332 超美洲豹的尾桨毂特点", "指点信标系统有几个指点信标台"]

demo = gr.ChatInterface(fn=echo,
                        examples=examples,
                        title="Echo Bot",
                        theme="soft")
demo.launch()

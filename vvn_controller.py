# rag_handler.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
Gemini_Api_key = os.getenv("GEMINI_API_KEY")

# Load and preprocess data
review_df = pd.read_csv("vvn_data.csv")

if 'ID' not in review_df.columns:
    raise KeyError("The column 'Product ID' does not exist in the DataFrame")

grouped_reviews_df = review_df.groupby('ID')
group_list = [group for _, group in grouped_reviews_df]
reviews_df = pd.concat(group_list)

def text_concat_info(row):
    _id = row["ID"]
    _name = row["Name"]
    _price = row["Price"]
    _category = row["Category"]
    _rating = row["rating"]
    _description = row["Description"]
    return 'Sản phẩm với ID: ' + str(_id) + ' có tên là ' + str(_name) + ', có mức giá: ' + str(
        _price) + ', thuộc loại : ' + str(_category) + ', có mức đánh giá là ' + str(
        _rating) + ', có giới thiệu là' + str(_description)

reviews_df["Result Text"] = reviews_df.apply(text_concat_info, axis=1)
reviews_df = reviews_df.dropna()
reviews_df = reviews_df.drop_duplicates(subset=["Result Text"])

text_splitter = RecursiveCharacterTextSplitter(
    separators=["."],
    chunk_size=400,
    chunk_overlap=0,
    is_separator_regex=False,
)

def split_into_chunks(text):
    docs = text_splitter.create_documents([text])
    text_chunks = [doc.page_content for doc in docs]
    return text_chunks

reviews_df["text_chunk"] = reviews_df["Result Text"].apply(split_into_chunks)
reviews_df = reviews_df.explode("text_chunk")
reviews_df["chunk_id"] = reviews_df.groupby(level=0).cumcount()

model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

text_chunks = reviews_df["text_chunk"].tolist()
text_chunk_vectors = model.encode(text_chunks, show_progress_bar=True)

def retrieve_relevant_documents(query, text_chunk_vectors, k):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], text_chunk_vectors)[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return reviews_df.iloc[top_k_indices]

def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

prompt_template = load_template('prompt_template.txt')

continue_prompt_template = load_template("continue_prompt_template.txt")

def create_prompt(query, k=5):
    relevant_rows = retrieve_relevant_documents(query, text_chunk_vectors, k)
    text_chunks = relevant_rows["text_chunk"].tolist()
    text_chunks_string = "\n".join(text_chunks)
    prompt = prompt_template
    prompt = prompt.replace("<documents>", text_chunks_string)
    prompt = prompt.replace("<query>", query)
    return prompt

def create_continue_prompt(query_history, query, k=5):
    relevant_rows = retrieve_relevant_documents(query, text_chunk_vectors, k)
    text_chunks = relevant_rows["text_chunk"].tolist()
    text_chunks_string = "\n".join(text_chunks)
    prompt = continue_prompt_template
    prompt = prompt.replace("<documents>", text_chunks_string)
    prompt = prompt.replace("<query>", query)
    query_history_str = "\n".join(query_history)
    print("Chat history: ", query_history_str)
    prompt = prompt.replace("<old_query>", query_history_str)
    return prompt

genai.configure(api_key=Gemini_Api_key)
geminiModel = genai.GenerativeModel('gemini-1.5-flash')

def generate(prompt):
    response = geminiModel.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=256,
            temperature=0.2
        )
    )
    return response.text

def send_new_chat(query):
    prompt = create_prompt(query)
    return generate(prompt)

def send_continue_chat(chat_history, query):
    history = []

    for chat in chat_history:
        chat_query = chat.content

        if chat.is_user:
            history.append("người dùng:" + chat_query)
        else:
            history.append("bạn:" + chat_query)

    prompt = create_continue_prompt(history, query)
    return generate(prompt)

# rag_handler.py
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd

load_dotenv()

Gemini_Api_key = os.getenv("GEMINI_API_KEY")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = Gemini_Api_key

genai.configure(api_key=Gemini_Api_key)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=250,
    max_retries=2,
)

# Load and preprocess data
review_df = pd.read_csv("vvn_data.csv")

if 'ID' not in review_df.columns:
    raise KeyError("The column 'Product ID' does not exist in the DataFrame")

grouped_reviews_df = review_df.groupby('ID')
group_list = [group for _, group in grouped_reviews_df]
reviews_df = pd.concat(group_list)

def load_prompt_template(file_path='prompt_template.txt'):
    with open(file_path, 'r') as file:
        return file.read()


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
    chunk_size=1000,
    chunk_overlap=200,
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
db = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 10}
)

prompt_template = load_prompt_template('langchain_prompt.txt')

chat_history = []

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    """Given a chat history between an AI chatbot and user
    that chatbot's message marked with [bot] prefix and user's message marked with [user] prefix,
    and given the latest user question.
    which might reference context in the chat history, 
    formulate a standalone question which can be understood 
    without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is. Write it in Vietnamese."""
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

question = "Tôi nặng 65kg, muốn tìm một chiếc áo thể thao"

ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
print(ai_msg_1["answer"])

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "Những sản phẩm này làm từ chất liệu gì?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])

def send_continue_chat(chat_history, query):
    history = []

    for chat in chat_history:
        chat_query = chat.content
        if chat.is_user:
            history.append("[bot] " + str(chat_query))
        else:
            history.append("[user] " + str(chat_query))

    response = rag_chain.invoke({"input": query, "chat_history": history})
    return response["answer"]

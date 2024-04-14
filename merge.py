import io
import base64
import pandas as pd
import streamlit as st
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader,PdfWriter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from PIL import Image
from io import StringIO
import configparser
import docx2txt
import sys

st.set_page_config(page_title="CHATBOT")

pg_bg_img="""
    <style>
    [data-testid="stAppViewContainer"]>.main{
background-color: #194bab;
background-image: linear-gradient(90deg, #194bab 0%, #6f6771 100%);

    }
    [data-testid = "stSidebar"]{
background-color: #124eb5;
background-image: linear-gradient(84deg, #124eb5 0%, #6f6771 100%);

     }
    </style>
    """
st.markdown(pg_bg_img,unsafe_allow_html=True)

api_key="AIzaSyAbzrXjPtxH9_i9Fbi0kHQOsE7juTdPMSs"
genai.configure(api_key="AIzaSyAbzrXjPtxH9_i9Fbi0kHQOsE7juTdPMSs")

generation_config = genai.GenerationConfig(
    temperature=0.9,
    top_p=1,
    top_k=1,
    max_output_tokens=2048,
)

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

lm1=GoogleGenerativeAI(model="gemini-1.0-pro",google_api_key=api_key)
llm0=genai.GenerativeModel("gemini-1.0-pro")

llm=GoogleGenerativeAI(model="gemini-pro-vision",google_api_key=api_key)
llm1=genai.GenerativeModel("gemini-pro-vision")


with st.sidebar:
        st.title("MENU :")

def extract_text_from_txt(file):
    text = file.read().decode('utf-8')
    return text

def extract_text_from_docx(file):
    # Creating a BytesIO object from the uploaded file
    docx_file_obj = io.BytesIO(file.read())
    # Extracting text from the Word file
    text = docx2txt.process(docx_file_obj)
    # Returning the extracted text
    return text

def app():
    st.header("CHAT WITH UR FILE!📄")
    st.subheader("Upload the file in Menu")
    uploaded_file = st.sidebar.file_uploader("Choose a file",type=["csv", "xlsx","pdf","png","jpeg","jpg","txt","docx"])
    if uploaded_file is not None:
                    file_extension = uploaded_file.name.split(".")[-1]
                    if file_extension in ["csv","xlsx"]:
                        data = pd.read_csv(uploaded_file)
                        display = create_pandas_dataframe_agent(lm1,data, verbose=True)
                        st.subheader("DATA DISPLAY :")
                        st.dataframe(data.head())
                        query = st.text_input("Enter a query : ")
                        if st.button("Execute"):
                            ans = display.run(query)
                            st.subheader("ANSWER :")
                            st.write(ans)
                            st.success("DONE")
                    elif file_extension in ["png","jpeg","jpg"]:
                        img=Image.open(uploaded_file)
                        query = st.text_input("Enter a query : ")
                        if st.button("Execute"):
                            res=llm1.generate_content([query,img])
                            st.subheader("ANSWER :")
                            st.write(res.text)

                    elif file_extension =="txt":
                        txt=extract_text_from_txt(uploaded_file)
                        prompt_template = """Answer the question as precise as possible using the provided context. If the answer is not contained in the context, say "answer not available in context" \n\n
                                                Context: \n {context}?\n
                                                Question: \n {question} \n
                                                Answer:
                                                """

                        prompt = PromptTemplate(
                            template=prompt_template, input_variables=["context", "question"]
                        )

                        query = st.text_input("Enter a query : ")
                        if st.button("Execute"):
                            res = llm0.generate_content([query, txt])
                            st.subheader("ANSWER :")
                            st.write(res.text)

                    elif file_extension == "docx":
                                texts = extract_text_from_docx(uploaded_file)
                                query = st.text_input("Enter a query : ")
                                if st.button("Execute"):
                                    res = llm0.generate_content([query, texts])
                                    st.subheader("ANSWER :")
                                    st.write(res.text)


                    elif file_extension =="pdf":
                        pdf_load = PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_load.pages:
                            text += page.extract_text()
                        prompt_template = """Answer the question as precise as possible using the provided context. If the answer is not contained in the context, say "answer not available in context" \n\n
                        Context: \n {context}?\n
                        Question: \n {question} \n
                        Answer:
                        """

                        prompt = PromptTemplate(
                            template=prompt_template, input_variables=["context", "question"]
                        )
                        stuff_chain = load_qa_chain(lm1, chain_type="stuff", prompt=prompt)

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
                        # context = "\n\n".join(str(p.page_content) for p in pages)
                        textss = text_splitter.split_text(text)
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                        vector_index = FAISS.from_texts(textss, embedding=embeddings)
                        query = st.text_input("Enter a query : ")
                        if st.button("Execute"):
                            ans = vector_index.similarity_search(query)
                            st.subheader("ANSWER :")
                            stuff_answer = stuff_chain(
                                {"input_documents": ans, "question": query}, return_only_outputs=True
                            )
                            print(stuff_answer)
                            st.write(stuff_answer["output_text"])
                            st.success("DONE")

                    else:
                        pass

    else:
                    st.warning("You need to upload a file!")

if __name__ == "__main__":
    app()

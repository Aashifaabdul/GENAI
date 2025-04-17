from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import   RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.llms import Ollama
import streamlit as st 
import time
import pickle
import os

st.title("News Research Tool!!")

st.sidebar.title("News ARticle URLs")
urls=[]
for i in range(2):
   url= st.sidebar.text_input(f"URL {i+1}")
   urls.append(url)

process_clicked=st.sidebar.button("Process URLs")
file_path="faiss_store.pkl"
main_placefolder=st.empty()

llm= Ollama(model="mistral", temperature=0.7)


if process_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading .....Started....")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['/n/n','/n',' '],
        chunk_size=1000
    )
    main_placefolder.text("Data Splitting .....Started....")
    docs=text_splitter.split_documents(data)
    embeddings=OllamaEmbeddings()
    vectors_store=FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Embedding vector.....Started Building.....")
    time.sleep(2)
    with open(file_path,"wb") as f:
        pickle.dump(vectors_store,f)


query=main_placefolder.text_input("Question:  ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            result=chain({"question":query},return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

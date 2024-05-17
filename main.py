import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

prompt_template = """
You are a professional doctor. Based on the blood statistics PDF uploaded, tell the patient their blood age (glycan age) and compare it to their chronological age. 
Provide the answer in this format:

Based on the information from your GlycanAge report, your biological age is [BIOLOGICAL_AGE] years. This is [DIFFERENCE] years [younger/older] than your chronological age, which is [CHRONOLOGICAL_AGE] years.

This indicates that your immune system and overall health are performing much [younger/older] than your actual age, likely due to factors such as lifestyle choices, diet, exercise, and potentially genetic advantages.
"""

prompt_template2 = """
You are a professional doctor. Explain what blood (glycan) age is as if you were explaining to a five-year-old.
"""

prompt_template3 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, explain what blood (glycan) age indicates about their life and how to maintain it.
"""

def get_pdf_content(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_content):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_content)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):

    st.session_state.chat_history += user_template.replace("{{MSG}}", user_question)

    response = st.session_state.conversation({'question': prompt_template})

    answer = response['answer']
    st.session_state.chat_history += bot_template.replace("{{MSG}}", answer)

    st.write(st.session_state.chat_history, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="DebateGPT", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""
    
    st.header("DebateGPT :books:")
    pdfs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
    processBtn = st.button("Process")
    if processBtn:
        if not pdfs: 
            st.error("Please upload at least one PDF file to proceed.")
        else:
            with st.spinner("Processing"):
                raw_content = get_pdf_content(pdfs)
                text_chunks = get_text_chunks(raw_content)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Processing complete! Ready to receive questions.")
    
    user_question = st.text_input("What's going on? (Type a topic or question here)")
    if user_question:
        handle_user_input(user_question)

    if st.session_state.conversation:
        st.write("DebateGPT is ready to respond to your questions.")

    with st.sidebar:
        st.subheader("Old blood tests:")

if __name__ == "__main__":
    main()

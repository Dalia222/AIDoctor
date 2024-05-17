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

# Initial prompts as described in the request
prompt_template0 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, tell the patient their blood age (glycan age) and compare it to their chronological age. 
Provide the answer in this format:

Hi there! I'm your AI Doctor, Based on the information from your GlycanAge report, your biological age is [BIOLOGICAL_AGE] years. This is [DIFFERENCE] years [younger/older] than your chronological age, which is [CHRONOLOGICAL_AGE] years.

This indicates that your immune system and overall health are performing much [younger/older] than your actual age, likely due to factors such as lifestyle choices, diet, exercise, and potentially genetic advantages.
"""
prompt_template1 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, answer the patient's questions {user_question} in a simple form
"""
prompt_template2 = """
You are a professional doctor. Explain what blood (glycan) age is as if you were explaining to a person with no medical background.
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
    if user_question == "":
        prompt = prompt_template0
    else:
        st.session_state.chat_history += user_template.replace("{{MSG}}", user_question)
        prompt = prompt_template1.format(user_question=user_question)

    response = st.session_state.conversation({'question': prompt})
    answer = response['answer']
    st.session_state.chat_history += bot_template.replace("{{MSG}}", answer)
    st.write(st.session_state.chat_history, unsafe_allow_html=True)

    st.session_state.follow_up = True

def handle_follow_up(answer):
    st.session_state.chat_history += user_template.replace("{{MSG}}", answer)
    
    if answer == "Yes":
        prompt = prompt_template3
    else:
        prompt = prompt_template2

    response = st.session_state.conversation({'question': prompt})
    bot_answer = response['answer']
    st.session_state.chat_history += bot_template.replace("{{MSG}}", bot_answer)
    
    st.session_state.follow_up = False  
    st.session_state.buttons_disabled = True
    st.write(st.session_state.chat_history, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="DoctorGPT", page_icon="🩺")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""
    if "follow_up" not in st.session_state:
        st.session_state.follow_up = False
    if "initial_prompt_done" not in st.session_state:
        st.session_state.initial_prompt_done = False
    if "buttons_disabled" not in st.session_state:
        st.session_state.buttons_disabled = False

    st.header("Your AI Doctor 🩺")
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
            
            handle_user_input("")
            st.session_state.initial_prompt_done = True

    if st.session_state.initial_prompt_done:
        if st.session_state.follow_up:
            st.write(bot_template.replace("{{MSG}}", "Are you familiar with blood age?"), unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            if col1.button("Yes", disabled=st.session_state.buttons_disabled):
                st.session_state.buttons_disabled = True
                handle_follow_up("Yes")
            if col2.button("No", disabled=st.session_state.buttons_disabled):
                st.session_state.buttons_disabled = True
                handle_follow_up("No")
            # Update the disabling state after handling the button press.
            st.session_state.buttons_disabled = True
        # else:
        #     user_question = st.text_input("What's going on? (Type a topic or question here)")
        #     if user_question:
        #         handle_user_input(user_question)

    if st.session_state.conversation and not st.session_state.follow_up:
        st.write("DoctorAi is ready to respond to your questions.")

    with st.sidebar:
        st.subheader("Old blood tests:")

if __name__ == "__main__":
    main()

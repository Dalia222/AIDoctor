import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template

# Load the CSS from the file
def load_css(file_name):
    with open(file_name, "r") as f:
        return f.read()

css = load_css("styles.css")

# Initial prompts as described in the request
prompt_template0 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, tell the patient their blood age (glycan age) and compare it to their chronological age. 
Provide the answer in this format:

Hi there! I'm your AI Health Assistant. Based on the information from your GlycanAge report, your biological age is [BIOLOGICAL_AGE] years. This is [DIFFERENCE] years [younger/older] than your chronological age, which is [CHRONOLOGICAL_AGE] years.

This indicates that your immune system and overall health are performing much [younger/older] than your actual age, likely due to factors such as lifestyle choices, diet, exercise, and potentially genetic advantages.
"""
prompt_template1 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, answer the patient's questions {user_question} in a simple form.
"""
prompt_template2 = """
You are a professional doctor. Explain what blood (glycan) age is as if you were explaining to a person with no medical background.
"""
prompt_template3 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, explain what blood (glycan) age indicates about their life and how to maintain it.
"""
prompt_template4 = """
You are a professional doctor. Explain what {blood_marker} is and what it indicates about the patient's health.
"""
prompt_template5 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, provide recommendations to improve the patient's {specific_metric}.
"""
prompt_template6 = """
You are a professional doctor. Based on the blood statistics PDF uploaded, provide general health advice to the patient.
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
    llm = ChatOpenAI(model="gpt-4o")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_initial_prompt():
    if "initial_prompt_done" not in st.session_state or not st.session_state.initial_prompt_done:
        prompt = prompt_template0
        response = st.session_state.conversation({'question': prompt})
        answer = response['answer']
        st.session_state.chat_history += bot_template.replace("{{MSG}}", answer)
        st.write(f'<div class="chat-container">{st.session_state.chat_history}</div>', unsafe_allow_html=True)
        st.session_state.initial_prompt_done = True
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
    st.session_state.follow_up_done = True  # Mark follow-up as done
    st.write(f'<div class="chat-container">{st.session_state.chat_history}</div>', unsafe_allow_html=True)

def handle_user_input(user_question):
    st.session_state.chat_history += user_template.replace("{{MSG}}", user_question)

    if "explain" in user_question.lower() and "what" in user_question.lower():
        blood_marker = user_question.split(" ")[-1]
        prompt = prompt_template4.format(blood_marker=blood_marker)
    elif "improve" in user_question.lower():
        specific_metric = user_question.split(" ")[-1]
        prompt = prompt_template5.format(specific_metric=specific_metric)
    elif "general health advice" in user_question.lower():
        prompt = prompt_template6
    else:
        prompt = prompt_template1.format(user_question=user_question)

    response = st.session_state.conversation({'question': prompt})
    answer = response['answer']
    st.session_state.chat_history += bot_template.replace("{{MSG}}", answer)
    st.write(f'<div class="chat-container">{st.session_state.chat_history}</div>', unsafe_allow_html=True)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Health Assistant", page_icon="🩺")
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)  # Ensure the CSS is applied early in the main function

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
    if "follow_up_done" not in st.session_state:
        st.session_state.follow_up_done = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    st.header("Your AI Health Assistant 🩺")
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
            
            # Extract filenames and append them to the session state
            filenames = [pdf.name for pdf in pdfs]
            st.session_state.uploaded_files.extend(filenames)
            
            st.success("Processing complete! Ready to receive questions.")
            
            handle_initial_prompt()

    if st.session_state.initial_prompt_done:
        if st.session_state.follow_up:
            st.write(bot_template.replace("{{MSG}}", "Are you familiar with blood age?"), unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            if col1.button("Yes", disabled=st.session_state.buttons_disabled):
                handle_follow_up("Yes")
            if col2.button("No", disabled=st.session_state.buttons_disabled):
                handle_follow_up("No")
            st.session_state.buttons_disabled = True
        
        if st.session_state.follow_up_done and not st.session_state.follow_up:
            user_question = st.text_input("What else do you need to know?")
            if user_question:
                handle_user_input(user_question)
            st.markdown("### Or select a predefined question:")
            if st.button("Explain what [blood_marker] indicates"):
                handle_user_input("Explain what [blood_marker] indicates")
            if st.button("What can I do to improve my [specific_metric]?"):
                handle_user_input("What can I do to improve my [specific_metric]?")
            if st.button("Can you give me some general health advice based on my blood report?"):
                handle_user_input("Can you give me some general health advice based on my blood report?")

    if st.session_state.conversation and not st.session_state.follow_up and not st.session_state.follow_up_done:
        st.write("DoctorGPT is ready to respond to your questions.")

    with st.sidebar:
        st.subheader("Uploads:")
        for file in st.session_state.uploaded_files:
            st.write(f"<p>  {file}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

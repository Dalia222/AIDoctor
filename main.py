import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub



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
        chunk_size =1000,
        chunk_overlap=200,
        length_function=len,
        )
    chunks=text_splitter.split_text(raw_content)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,         
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="DebateGPT", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None 
    if "chat_history" not in st.session_state:  
        st.session_state.chat_history = None
    st.header("DebateGPT :books:")
    user_question = st.text_input("What's going on? (Type a topic or question here)")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello?"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hi There!"),unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Library: ")
        pdfs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        processBtn = st.button("Process")
        if processBtn:
            with st.spinner("Processing"):
                # get the pdf text (Raw Content)
                raw_content = get_pdf_content(pdfs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_content)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
            
            st.success("Processing complete! Ready to receive questions.") # Show this instead of the actual processed data

    if st.session_state.conversation:
        st.write("DebateGPT is ready to respond to your questions.")


if __name__ == "__main__":
    main()
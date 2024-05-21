import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Set up Streamlit app
st.title("Jet HR Writing Assistant")

# Get API key from user
api_key = st.text_input("Enter your OpenAI API key", type="password")

# Get PDF file from user
pdf_file = st.file_uploader("Carica i PDF con le informazioni", type="pdf")

# Get model and temperature from user
model_name = st.selectbox("Select a model", ["gpt-3.5-turbo", "gpt-4o"])
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

if pdf_file is not None and api_key:
    try:
        # Save uploaded file to a temporary path
        tmp_file_path = "/tmp/uploaded_pdf.pdf"
        with open(tmp_file_path, "wb") as f:
            f.write(pdf_file.getvalue())

        # Load PDF and split into chunks
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.from_documents(texts, embeddings)

        # Set up question answering chain
        llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model_name=model_name)

        chain = load_qa_chain(llm, chain_type="stuff")

        # Get user question
        prompt = """Sei un esperto agente del support clienti di Jet HR. Rispondi in modo chiaro, completo e preciso alle domande dei clienti in base alle informazioni che hai disponibili. Se la domanda non c'entra con JetHR, rispondi che non puoi rispondere a quella domanda. Rispondi sempre in Italiano. Struttura la risposta come fosse una mail cordiale ma non troppo formale.
        Importante: se non hai informazioni, NON INVENTARE UNA RISPOSTA. Di semplicemente che non hai gli elementi per rispondere. 
        Ecco la domanda:
        """
        query = st.text_input("Inserisci il ticket")
        full_prompt = prompt + query

        if query:
            # Find relevant documents and answer question
            docs = db.similarity_search(query)
            answer = chain.run(input_documents=docs, question=full_prompt)
            st.write(answer)

    except ImportError as e:
        st.error(f"An error occurred: {e}. Please ensure your OpenAI API key is correct and all dependencies are installed.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

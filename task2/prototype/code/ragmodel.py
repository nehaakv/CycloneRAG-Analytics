import os
import torch
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings

warnings.filterwarnings("ignore")

#Configuration

DOCS_PATH = "C:\\Users\\nehak\\OneDrive\\Desktop\\cse\\exactspace\\task2\\prototype\\docs\\drive-download-20251005T121415Z-1-001"
DB_PATH = 'vectorstore/'
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
LLM_MODEL_NAME = 'google/flan-t5-base' 

def create_vector_db():
    """
    This function processes the documents in the DOCS_PATH,
    creates text embeddings, and saves them to a local FAISS vector store.
    """
    if os.path.exists(DB_PATH):
        print("Vector database already exists.")
        return

    print("Creating a new vector database from the documents...")
    loader = DirectoryLoader(DOCS_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError("No documents found in the 'docs' directory. Please add your PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(text_chunks)} chunks.")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # Create the vector store using FAISS from the text chunks and embeddings
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"Vector database created and saved at {DB_PATH}")
    return

#Inference Pipeline: Setup the RAG Chain
def create_rag_chain():
  
  
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector database not found at {DB_PATH}. Please run the script once to create it.")
        
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 4}) 
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    
    # Create a pipeline for text generation
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1, 
        top_p=0.95
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create a prompt template. 
    prompt_template = """
    Use the following pieces of context from the technical manuals to answer the question at the end.
    If you don't know the answer from the context provided, just say that you don't know.
    Provide a concise and helpful response, citing the source document for each claim.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

if __name__ == "__main__":
    
    create_vector_db()
    qa_chain = create_rag_chain()

    print("\n--- RAG System Ready ---")
    print("Ask a question about your cyclone separator documents. Type 'exit' to quit.")
    
    # Example questions 
    print("\nExample questions:")
    print("- What are the daily maintenance checks for a cyclone separator?")
    print("- How do I troubleshoot a blockage in the dust discharge?")
    print("- What is the function of the vortex finder in a Parker DustHog collector?")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break
        result = qa_chain.invoke({"query": query})
        print("\nAnswer:")
        print(result['result'])
        print("\nSources Used:")
        for doc in result['source_documents']:
            print(f"- {os.path.basename(doc.metadata['source'])} (Page: {doc.metadata.get('page', 'N/A')})")
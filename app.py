import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ========== LLM LOADING (CACHED) ==========
@st.cache_resource
def load_llm():
    """
    Loads the Flan-T5 Large model and wraps it in a LangChain-compatible pipeline.
    This is cached so it only loads once per server session.
    """
    st.info("Loading the AI model... This may take a few minutes the first time.")
    
    model_name = "google/flan-t5-large"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    # Wrap in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm

# Load the LLM at startup
llm = load_llm()

# ========== MAIN APPLICATION ==========
def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat with Your Data",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Chat with Your Data - RAG Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # ========== SIDEBAR: FILE UPLOAD & PROCESSING ==========
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help="Select a PDF document to analyze"
        )
        
        # Process the uploaded file
        if uploaded_file is not None:
            # Check if this is a new file or already processed
            if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
                
                with st.spinner("🔄 Processing your document..."):
                    try:
                        # Save uploaded file to a temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # ===== PHASE 2: INGESTION PIPELINE =====
                        
                        # Step 1: LOAD the PDF
                        loader = PyPDFLoader(tmp_file_path)
                        documents = loader.load()
                        
                        # Step 2: SPLIT into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        texts = text_splitter.split_documents(documents)
                        
                        # Step 3: EMBED the chunks using sentence-transformers
                        embeddings = HuggingFaceEmbeddings(
                            model_name="all-MiniLM-L6-v2"
                        )
                        
                        # Step 4: STORE in vector database (FAISS)
                        vectorstore = FAISS.from_documents(texts, embeddings)
                        
                        # ===== END OF INGESTION =====
                        
                        # Clean up temporary file
                        os.remove(tmp_file_path)
                        
                        # Store in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.last_file = uploaded_file.name
                        
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                        st.info(f"📊 Created {len(texts)} text chunks for analysis")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)
            else:
                st.success(f"✅ Using loaded document: {uploaded_file.name}")
        
        # Display info about the system
        st.markdown("---")
        st.markdown("### 🔧 System Info")
        st.markdown("**Embedding Model:** all-MiniLM-L6-v2")
        st.markdown("**LLM:** Flan-T5 Large")
        st.markdown("**Vector DB:** FAISS")
    
    # ========== MAIN AREA: CHAT INTERFACE ==========
    st.header("💬 Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! 👋 Please upload a PDF document using the sidebar to get started."
        }]
    
    # Display all chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if vectorstore is ready
        if "vectorstore" not in st.session_state:
            response = "⚠️ Please upload a document first using the sidebar."
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:
            # ===== PHASE 3: RAG QUERY PIPELINE (MODERN API) =====
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Step 1: RETRIEVE relevant chunks
                        vectorstore = st.session_state.vectorstore
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
                        )
                        
                        # Step 2: CREATE the prompt template
                        system_prompt = (
                            "You are a helpful assistant that answers questions based on the provided context. "
                            "Use the given context to answer the question. "
                            "If you don't know the answer based on the context, say you don't know. "
                            "Keep the answer concise and relevant.\n\n"
                            "Context: {context}"
                        )
                        
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        
                        # Step 3: CREATE the document chain
                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        
                        # Step 4: CREATE the retrieval chain
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        # Step 5: GENERATE response
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        source_docs = response.get("context", [])
                        
                        # ===== END OF RAG =====
                        
                        # Display the answer
                        st.markdown(answer)
                        
                        # Optional: Show source chunks in an expander
                        if source_docs:
                            with st.expander("📚 View Source Chunks"):
                                for i, doc in enumerate(source_docs, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(doc.page_content[:300] + "...")
                                    st.markdown("---")
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    
    # Add a button to clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat history cleared. Ask me anything about your document!"
        }]
        st.rerun()

# ========== RUN THE APP ==========
if __name__ == "__main__":
    main()
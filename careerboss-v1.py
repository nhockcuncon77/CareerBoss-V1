import streamlit as st
import time
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 🔐 Set credentials
PINECONE_API_KEY = "pcsk_6MvBEW_Lzrqekycsbj5snYqthzSVTc8aiUvCUrjQ3SQ9nNPn4WvVPAcJBHVakncsPf4vH"
OPENAI_API_KEY = "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"
PINECONE_INDEX_NAME = "career-boss-v1"
PINECONE_REGION = "us-east-1"

# --- 📦 Initialize Pinecone and LangChain ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

# --- Streamlit page config ---
st.set_page_config(page_title="Career Coach AI", page_icon="🧠")
st.title("💼 Career Advice Assistant")

# --- Session-based memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful career coach. Use the documents stored in Pinecone to support your advice.")
    ]

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# --- Get new user input ---
user_input = st.chat_input("Ask your career question...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # --- Retrieve relevant documents ---
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.3}
    )
    docs = retriever.invoke(user_input)

    # Combine docs into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create strong system prompt with context
    system_prompt = f"""
You are a professional career coach helping people stuck in jobs that don't align with their passion or purpose.

Use the following CONTEXT from our knowledge base to support your answers.

CONTEXT:
{context}

Instructions:
- Always be encouraging and empathetic.
- Reference examples from the documents if possible.
- Never make up facts that aren't in the context.
- If you don't know something, say "I don't know, but I can help you explore it."
- If a user expresses frustration or confusion, acknowledge it and provide actionable steps.
"""

    # Update system message
    st.session_state.messages = [
        msg for msg in st.session_state.messages if not isinstance(msg, SystemMessage)
    ]
    st.session_state.messages.insert(0, SystemMessage(content=system_prompt))

    # --- Chat with OpenAI ---
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = llm.invoke(st.session_state.messages).content
            time.sleep(1.2)
            st.markdown(reply)

    st.session_state.messages.append(AIMessage(content=reply))

    # --- Optional: Debug context
    with st.expander("📄 Retrieved Documents"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Doc {i+1}**")
            st.write(doc.page_content)
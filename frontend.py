import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🧠 DocuMind AI")
st.caption("Local RAG Engine (Phi-3-mini + pgvector + Redis Cache)")

# FastAPI Backend Endpoint
BACKEND_URL = "http://127.0.0.1:8000"

# --- SIDEBAR: Document Upload Section ---
with st.sidebar:
    st.header("📄 Document Ingestion")
    st.markdown("Upload your PDFs or text documents to index them into pgvector.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("🚀 Upload & Process", use_container_width=True):
            with st.spinner("Uploading document to backend..."):
                try:
                    # Send file as multipart/form-data to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ '{uploaded_file.name}' uploaded and queued for processing!")
                    else:
                        st.error(f"❌ Error ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Connection failed! Check if FastAPI is running.")

# --- MAIN CHAT INTERFACE ---
# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata only for assistant messages
        if message["role"] == "assistant":
            if message.get("sources"):
                st.caption(f"📚 **Sources:** {', '.join(message['sources'])}")
            
            badge = "⚡ Cache Hit (Redis)" if message.get("cached", False) else "🧠 Fresh LLM Generation (GPU)"
            st.caption(f"Status: {badge}")

# User Input
if prompt := st.chat_input("Ask anything from your documents..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant"):
        with st.spinner("Generating answer via local GPU..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    is_cached = data.get("cached", False)
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources
                    if sources:
                        st.caption(f"📚 **Sources:** {', '.join(sources)}")
                    
                    # Display Cache Badge
                    badge = "⚡ Cache Hit (Redis)" if is_cached else "🧠 Fresh LLM Generation (GPU)"
                    st.caption(f"Status: {badge}")
                    
                    # Save to State
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "cached": is_cached
                    })
                else:
                    st.error(f"Backend Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Connection failed! Make sure your FastAPI backend is running on port 8000.")
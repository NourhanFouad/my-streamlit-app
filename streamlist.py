import streamlit as st
from app import (
    authenticate_gdrive,
    process_drive_files,
    search_qdrant,
    generate_answer,
    init_qdrant,
    init_gemini
)

st.set_page_config(page_title="Smart Document Search System", layout="wide")
st.title("Smart Document Search with Gemini and Google Drive")

# Initialize session state variables
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "drive_service" not in st.session_state:
    st.session_state.drive_service = None
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "gemini_initialized" not in st.session_state:
    st.session_state.gemini_initialized = False
if "files_indexed" not in st.session_state:
    st.session_state.files_indexed = False

# Authentication section
if not st.session_state.is_authenticated:
    st.header("Google Drive Login")
    if st.button("Login"):
        with st.spinner("Authenticating..."):
            service = authenticate_gdrive()
            if service:
                st.session_state.drive_service = service
                st.session_state.is_authenticated = True
                st.success("Logged in successfully")
            else:
                st.error("Login failed, please try again")
else:
    st.sidebar.header("Account Info")
    st.sidebar.write("Connected to Google Drive")

    # Initialize Qdrant client once
    if st.session_state.qdrant_client is None:
        st.session_state.qdrant_client = init_qdrant()

    # Initialize Gemini once
    if not st.session_state.gemini_initialized:
        try:
            init_gemini()
            st.session_state.gemini_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}")
            st.stop()

    # Process and index files once
    if not st.session_state.files_indexed:
        with st.spinner("Indexing files from Google Drive..."):
            try:
                process_drive_files(st.session_state.drive_service, st.session_state.qdrant_client)
                st.session_state.files_indexed = True
                st.success("Files indexed successfully")
            except Exception as e:
                st.error(f"Error indexing files: {e}")
                st.stop()

    # Search interface
    st.header("Search Your Documents")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Searching and generating answer..."):
            results = search_qdrant(st.session_state.qdrant_client, query)
            if results:
                context_texts = "\n\n".join(
                    [f"File: {r.payload['name']}\n{r.payload['content']}" for r in results]
                )
                answer = generate_answer(query, context_texts)
                st.subheader("Answer:")
                st.write(answer)

                st.subheader("Sources:")
                for idx, res in enumerate(results, 1):
                    st.write(f"{idx}. {res.payload['name']} (Score: {res.score:.3f})")
            else:
                st.warning("No relevant information found.")

    if st.sidebar.button("Logout"):
        keys_to_clear = [
            "is_authenticated",
            "drive_service",
            "qdrant_client",
            "gemini_initialized",
            "files_indexed",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.experimental_rerun()

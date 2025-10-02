import os
import io
import pickle
import warnings
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from qdrant_client.http.models import PointStruct
import google.generativeai as genai
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
COLLECTION_NAME = "drive_docs"
VECTOR_SIZE = 768
DISTANCE_METRIC = Distance.COSINE

def authenticate_gdrive():
    creds = None
    token_file = "token.pickle"
    try:
        if os.path.exists(token_file):
            with open(token_file, "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=8000, open_browser=True, prompt='consent')
            with open(token_file, "wb") as token:
                pickle.dump(creds, token)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def init_qdrant():
    client = QdrantClient(":memory:")  # or your Qdrant server URL
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC)
        )
    return client

def init_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    return genai

def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'])
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.zeros(VECTOR_SIZE)

def chunk_text(text, max_chars=30000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def read_file(file_id, mime_type, service):
    try:
        if mime_type.startswith("application/vnd.google-apps"):
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        else:
            request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        content = fh.getvalue().decode("utf-8", errors='ignore').strip()
        return content
    except Exception as e:
        print(f"Error reading file {file_id}: {e}")
        return ""

def process_drive_files(service, qdrant_client):
    try:
        results = service.files().list(
            pageSize=100,
            fields="files(id, name, mimeType)",
            q="trashed=false"
        ).execute()

        items = results.get("files", [])
        allowed_types = [
            'application/vnd.google-apps.document',
            'application/vnd.google-apps.spreadsheet',
            'application/vnd.google-apps.presentation',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain'
        ]

        for idx, item in enumerate(items, 1):
            if item.get("mimeType") not in allowed_types:
                continue

            file_id = item.get("id")
            file_name = item.get("name", "unnamed")
            print(f"Processing {idx}/{len(items)}: {file_name}")

            if not file_id:
                print(f"Skipping file with no ID: {file_name}")
                continue

            content = read_file(file_id, item.get("mimeType"), service)
            if len(content) < 20:
                print(f"Skipping small file: {file_name}")
                continue

            chunks = chunk_text(content)
            embeddings = []
            for chunk in chunks:
                embedding = get_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding)

            if not embeddings:
                print(f"No valid embeddings generated for: {file_name}")
                continue

            avg_embedding = np.mean(embeddings, axis=0).tolist()
            point_id = abs(hash(file_id)) % (10**18)

            points = [
                PointStruct(
                    id=point_id,
                    vector=avg_embedding,
                    payload={
                        "name": file_name,
                        "content": content[:3000]  # Increased snippet size for better context
                    }
                )
            ]

            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"Processed: {file_name}")

    except Exception as e:
        print(f"Error processing files: {e}")
        raise

def search_qdrant(qdrant_client, query, top_k=3):
    try:
        query_embedding = get_embedding(query)
        if query_embedding is None or np.all(query_embedding == 0):
            print("Invalid query embedding")
            return []

        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
        return search_result

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_answer(query, context):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
Based on the following context, answer the question clearly and precisely.
If the answer is not found in the context, say that the information is not available.

Context:
{context[:8000]}

Question: {query}
"""
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text.strip():
            return response.text.strip()
        else:
            return "Could not generate a clear answer."
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."

def main():
    print("Initializing Drive2Chat...")

    print("Authenticating with Google Drive...")
    drive_service = authenticate_gdrive()
    if not drive_service:
        print("Failed to authenticate with Google Drive")
        return

    print("Initializing Qdrant...")
    qdrant_client = init_qdrant()

    print("Initializing Gemini...")
    try:
        init_gemini()
    except Exception as e:
        print(f"Gemini initialization error: {e}")
        return

    print("\nProcessing files from Google Drive...")
    process_drive_files(drive_service, qdrant_client)

    print("\nReady! Ask me anything about your documents (or type 'quit' to exit):")
    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ('quit', 'exit'):
                print("Goodbye!")
                break

            if not query:
                continue

            results = search_qdrant(qdrant_client, query)
            if not results:
                print("No relevant information found.")
                continue

            context = "\n\n".join([f"From {r.payload['name']}:\n{r.payload['content']}" for r in results])

            print("\nSearching for information...")
            answer = generate_answer(query, context)
            print(f"\nAnswer:\n{answer}")

            print("\nSources:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.payload['name']} (score: {result.score:.3f})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()

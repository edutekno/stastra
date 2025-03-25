import streamlit as st
from astrapy import DataAPIClient
import PyPDF2
import uuid
import struct  # Tambahkan import struct

try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"Gagal mengimpor HuggingFaceEmbeddings: {e}. Pastikan 'langchain' dan 'sentence-transformers' terinstal.")
    st.stop()

# Konfigurasi AstraDB
ASTRA_TOKEN  = "AstraCS:MDMewQNUDbBKAnhWQloWFJAd:1598542746211376ce411be39fcfae4ba270721cb90d013eefccd44031dbf600"
ASTRA_API_ENDPOINT = "https://770919ae-7b86-4f0f-acfb-c77411020455-us-east-2.apps.astra.datastax.com"
KEYSPACE = "default_keyspace"
COLLECTION_NAME = "pdf_documents"

# Inisialisasi klien AstraDB
client = DataAPIClient(ASTRA_TOKEN)
db = client.get_database_by_api_endpoint(ASTRA_API_ENDPOINT, keyspace=KEYSPACE)

try:
    collection = db.create_collection(COLLECTION_NAME, dimension=384)
except Exception as e:
    collection = db.get_collection(COLLECTION_NAME)

# Inisialisasi embeddings
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Gagal inisialisasi embeddings: {e}")
    st.stop()

# Fungsi untuk ekstrak teks dari PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Fungsi untuk menyimpan teks dan vektor ke AstraDB
def store_in_astra_db(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    for chunk in chunks:
        embedding = embeddings_model.embed_query(chunk)
        # Konversi vektor ke binary
        embedding_binary = struct.pack('f' * len(embedding), *embedding)
        doc_id = str(uuid.uuid4())
        collection.insert_one({
            "_id": doc_id,
            "text": chunk,
            "embedding": embedding_binary  # Simpan sebagai binary
        })

# Fungsi untuk mencari jawaban
def search_answer(question):
    question_embedding = embeddings_model.embed_query(question)
    # Konversi vektor pertanyaan ke binary
    question_embedding_binary = struct.pack('f' * len(question_embedding), *question_embedding)
    
    result = collection.find_one(
        sort={"embedding": {"$vector": question_embedding_binary}},  # Gunakan binary
        projection={"text": 1}
    )
    return result["text"] if result else "Maaf, saya tidak menemukan jawaban yang relevan."

# Streamlit UI
st.title("Chat2PDF - Tanyakan Apa Saja dari PDF Anda")
uploaded_file = st.file_uploader("Unggah file PDF", type="pdf")

if uploaded_file:
    st.write("PDF berhasil diunggah. Sedang memproses...")
    try:
        text = extract_text_from_pdf(uploaded_file)
        store_in_astra_db(text)
        st.write("PDF telah diproses dan disimpan ke database.")
    except Exception as e:
        st.error(f"Error saat memproses PDF: {e}")

question = st.text_input("Masukkan pertanyaan Anda tentang isi PDF:")
if question and uploaded_file:
    try:
        answer = search_answer(question)
        st.write("Jawaban:", answer)
    except Exception as e:
        st.error(f"Error saat mencari jawaban: {e}")

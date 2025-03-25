import streamlit as st
from astrapy import DataAPIClient
from langchain.embeddings import HuggingFaceEmbeddings
import PyPDF2
import uuid

# Konfigurasi AstraDB
ASTRA_TOKEN  = "AstraCS:MDMewQNUDbBKAnhWQloWFJAd:1598542746211376ce411be39fcfae4ba270721cb90d013eefccd44031dbf600"  # Ganti dengan token Anda
ASTRA_API_ENDPOINT = "https://770919ae-7b86-4f0f-acfb-c77411020455-us-east-2.apps.astra.datastax.com"  # Ganti dengan API Endpoint Anda
KEYSPACE = "default_keyspace"  # Ganti jika Anda menggunakan keyspace lain
COLLECTION_NAME = "pdf_documents"

# Inisialisasi klien AstraDB
client = DataAPIClient(ASTRA_TOKEN)
db = client.get_database_by_api_endpoint(ASTRA_API_ENDPOINT, keyspace=KEYSPACE)

# Buat atau akses koleksi
try:
    collection = db.create_collection(COLLECTION_NAME, dimension=384)  # 384 adalah dimensi dari MiniLM-L6-v2
except:
    collection = db.get_collection(COLLECTION_NAME)

# Inisialisasi embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Fungsi untuk ekstrak teks dari PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Fungsi untuk menyimpan teks dan vektor ke AstraDB
def store_in_astra_db(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Pecah teks menjadi potongan kecil
    for chunk in chunks:
        embedding = embeddings_model.encode(chunk).tolist()
        doc_id = str(uuid.uuid4())  # ID unik untuk setiap dokumen
        collection.insert_one({"_id": doc_id, "text": chunk, "embedding": embedding})

# Fungsi untuk mencari jawaban
def search_answer(question):
    question_embedding = embeddings_model.encode(question).tolist()
    result = collection.find_one(
        sort={"embedding": {"$vector": question_embedding}},
        projection={"text": 1}
    )
    return result["text"] if result else "Maaf, saya tidak menemukan jawaban yang relevan."

# Streamlit UI
st.title("Chat2PDF - Tanyakan Apa Saja dari PDF Anda")

# Unggah PDF
uploaded_file = st.file_uploader("Unggah file PDF", type="pdf")

if uploaded_file:
    st.write("PDF berhasil diunggah. Sedang memproses...")
    text = extract_text_from_pdf(uploaded_file)
    store_in_astra_db(text)
    st.write("PDF telah diproses dan disimpan ke database.")

# Input pertanyaan
question = st.text_input("Masukkan pertanyaan Anda tentang isi PDF:")

if question and uploaded_file:
    answer = search_answer(question)
    st.write("Jawaban:", answer)
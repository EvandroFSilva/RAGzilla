import os
import json
import pdfplumber
import numpy as np
import faiss
import spacy
from dotenv import load_dotenv
from openai import OpenAI

# ==============================================
# 1. Configuração do ambiente
# ==============================================
load_dotenv()
client = OpenAI()

PDF_FOLDER = r"D:\Biopark\4p\PI3\documents\raw"  # <<< ALTERE AQUI

# ==============================================
# 2. Extrair texto dos PDFs
# ==============================================
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_documents(pdf_folder: str):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            full_path = os.path.join(pdf_folder, file)
            text = extract_text_from_pdf(full_path)
            documents.append({"file": file, "text": text})
    return documents

# ==============================================
# 3. Chunking de texto
# ==============================================
def chunk_text(text, chunk_size=500):
    """Divide texto em pedaços de 'chunk_size' palavras."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ==============================================
# 4. NER com spaCy
# ==============================================
nlp = spacy.load("pt_core_news_sm")

def extract_entities(text: str):
    spacy_doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
    return entities

# ==============================================
# 5. Embeddings OpenAI
# ==============================================
def gerar_embedding(texto: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return np.array(resp.data[0].embedding, dtype="float32")

# ==============================================
# 6. Criar índice FAISS com chunking
# ==============================================
def build_faiss_index(documents, chunk_size=500):
    texts = []
    metadata = []

    for doc in documents:
        doc_chunks = chunk_text(doc["text"], chunk_size)
        for chunk in doc_chunks:
            texts.append(chunk)
            metadata.append({"file": doc["file"], "text": chunk})

    embeddings = [gerar_embedding(t) for t in texts]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return index, embeddings, metadata

# ==============================================
# 7. Consulta RAG
# ==============================================
def rag_query(query: str, index, metadata, top_k: int = 2) -> str:
    query_emb = gerar_embedding(query)
    distances, indices = index.search(np.array([query_emb]), top_k)

    context = "\n\n".join([metadata[i]["text"] for i in indices[0]])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente que responde com base em documentos PDF fornecidos."},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# ==============================================
# 8. Execução principal
# ==============================================
if __name__ == "__main__":
    print("📂 Carregando documentos...")
    documents = load_documents(PDF_FOLDER)

    print("📑 Construindo índice vetorial com chunking...")
    index, embeddings, metadata = build_faiss_index(documents, chunk_size=500)

    # Exemplo 1: Extrair entidades do primeiro PDF
    print("\n=== ENTIDADES DO PRIMEIRO DOCUMENTO ===")
    entities = extract_entities(documents[0]["text"][:3000])
    print(entities)

    # Exemplo 2: Perguntar via RAG
    print("\n=== RESPOSTA VIA RAG ===")
    pergunta = "Quais organizações aparecem nos documentos?"
    resposta = rag_query(pergunta, index, metadata)
    print(resposta)

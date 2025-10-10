"""
RAG para contratos on-prem
Pipeline mínima para ingestão de PDFs, chunking inteligente,
embeddings locais com sentence-transformers, index FAISS persistente,
e função de consulta RAG que usa a OpenAI (sua chave em .env) para gerar
respostas/sumários/extrações a partir dos trechos recuperados.

Executar: python rag_contratos_onprem.py --ingest   # para indexar
Executar: python rag_contratos_onprem.py --query "Pergunta do advogado"

Observações:
  - Embeddings: usamos modelo multilingual local (boa cobertura PT-BR)
  - Index: FAISS (persistente no disco via gravação do índice + metadata.jsonl)
  - LLM: usa OpenAI apenas para geração (RAG). 

"""

import os
import argparse
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import openai

# -------------------- Config ------------------
load_dotenv()
PDF_FOLDER = Path(os.getenv("PDF_FOLDER", "pdfs"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Persistência
INDEX_DIR = Path("vector_index")
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.jsonl"
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  

# Chunking params
CHUNK_CHAR_SIZE = 3000  
CHUNK_OVERLAP = 400

# Retrieval
TOP_K = 6

# -------------------- Utilidades --------------------

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# -------------------- Extração de texto --------------------

def extract_text_from_pdf(path: Path) -> str:
    """Extrai texto de PDF assumindo que é texto (não-OCR).
    """
    doc = fitz.open(str(path))
    parts = []
    for p in doc:
        txt = p.get_text("text")
        parts.append(txt)
    doc.close()
    return "\n".join(parts)


# -------------------- Chunking inteligente --------------------
import re

def smart_chunk(text: str, chunk_size: int = CHUNK_CHAR_SIZE, overlap: int = CHUNK_OVERLAP):
    """Tenta dividir por cláusulas/headings. Se não houver, usa chunk fixo com overlap.
    Retorna lista de (chunk_text, start_char, end_char)
    """
    # Normaliza quebras
    t = text.replace('\r', '')

    # Tentativa 1: dividir por headings comuns em contratos (CLÁUSULA, Cláusula, Art., §, PARÁGRAFO)
    headings_pattern = re.compile(r"(?=\n\s*(CLÁUSULA\b|Cláusula\b|ARTIGO\b|Artigo\b|Art\.|PARÁGRAFO\b|Parágrafo\b|§))", re.IGNORECASE)
    parts = [s.strip() for s in headings_pattern.split(t) if s and not s.isspace()]

    # Se headings achar muitas partes razoáveis, recombinar para chunks
    if len(parts) >= 4:
        # Recombina até atingir chunk_size
        chunks = []
        cur = ""
        cur_start = 0
        i = 0
        for p in parts:
            if not p.strip():
                continue
            if len(cur) + len(p) <= chunk_size:
                if cur == "":
                    cur_start = i
                cur += "\n\n" + p
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = p
            i += len(p)
        if cur:
            chunks.append(cur.strip())
        # convert to tuples with character offsets (approximate)
        out = []
        pos = 0
        for c in chunks:
            start = t.find(c, pos)
            if start == -1:
                start = pos
            end = start + len(c)
            out.append((c, start, end))
            pos = end
        return out

    # Fallback: chunk fixo com overlap
    out = []
    i = 0
    L = len(t)
    while i < L:
        end = min(i + chunk_size, L)
        chunk = t[i:end]
        out.append((chunk, i, end))
        i = max(end - overlap, end)
    return out


# -------------------- Embeddings (local) --------------------

def init_embedding_model(model_name=EMB_MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model


# -------------------- Index (FAISS) --------------------

def ensure_index_dir():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


class SimpleFAISSIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas = [] 

    def add(self, vectors, metadatas):
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def save(self, path_index: Path, path_meta: Path):
        faiss.write_index(self.index, str(path_index))
        with open(path_meta, 'w', encoding='utf-8') as f:
            for m in self.metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

    @classmethod
    def load(cls, path_index: Path, path_meta: Path):
        idx = faiss.read_index(str(path_index))
        dim = idx.d
        obj = cls(dim)
        obj.index = idx
        obj.metadatas = []
        if path_meta.exists():
            with open(path_meta, 'r', encoding='utf-8') as f:
                for line in f:
                    obj.metadatas.append(json.loads(line))
        return obj

    def search(self, vector, top_k=TOP_K):
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(vector, top_k)
        results = []
        for dist_list, idx_list in zip(D, I):
            for dist, idx in zip(dist_list, idx_list):
                if idx < 0 or idx >= len(self.metadatas):
                    continue
                m = self.metadatas[idx]
                results.append({"score": float(dist), "metadata": m})
        return results


# -------------------- Ingestão --------------------

def ingest_all_pdfs(pdf_folder: Path, model: SentenceTransformer):
    ensure_index_dir()
    metadata_items = []
    vectors = []

    for pdf_path in tqdm(list(pdf_folder.glob("*.pdf")), desc="PDFs"):
        doc_id = pdf_path.stem
        file_hash = sha1_of_file(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        chunks = smart_chunk(text)
        for i, (chunk_text, start, end) in enumerate(chunks):
            emb = model.encode(chunk_text)
            vectors.append(emb.astype('float32'))
            metadata_items.append({
                "doc_id": doc_id,
                "file": str(pdf_path.name),
                "chunk_index": i,
                "start_char": start,
                "end_char": end,
                "text_snippet": chunk_text[:2000],
                "file_hash": file_hash,
            })

    import numpy as np
    if not vectors:
        print("Nenhum chunk gerado — verifique a pasta de PDFs")
        return

    arr = np.vstack(vectors)
    dim = arr.shape[1]
    faiss_idx = SimpleFAISSIndex(dim)
    faiss_idx.add(arr, metadata_items)
    faiss_idx.save(INDEX_FILE, METADATA_FILE)
    print("Index salvo em", INDEX_DIR)


# -------------------- Query / RAG --------------------

def load_index():
    ensure_index_dir()
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Index não encontrado. Rode --ingest primeiro.")
    idx = SimpleFAISSIndex.load(INDEX_FILE, METADATA_FILE)
    return idx


def call_openai_chat(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = openai.ChatCompletion.create(model=model_name, messages=messages, max_tokens=800)
    return resp['choices'][0]['message']['content']


PROMPT_TEMPLATE_SUMMARY = (
    "Você é um assistente jurídico. Use SOMENTE os trechos abaixo para responder."
    " Retorne JSON com: {{'summary':..., 'doc_citations': [{{'doc_id':..., 'chunk_index':..., 'text_excerpt':...}}]}}."
)

PROMPT_TEMPLATE_QA = (
    "Você é um assistente jurídico. Você receberá TRECHOS de contratos e uma PERGUNTA."
    " Use apenas as informações dos trechos. Retorne resposta objetiva em Português e inclua a lista de citações usadas"
)


def query_rag(question: str, model_embed: SentenceTransformer, top_k: int = TOP_K):
    idx = load_index()
    qv = model_embed.encode(question).astype('float32')
    import numpy as np
    qv = np.expand_dims(qv, axis=0)
    hits = idx.search(qv, top_k=top_k)
    # montar contexto
    context_items = []
    for h in hits:
        m = h['metadata']
        context_items.append({
            'doc_id': m['doc_id'],
            'chunk_index': m['chunk_index'],
            'score': h['score'],
            'text_snippet': m['text_snippet']
        })

    # construir prompt com trechos
    context_text = "\n\n---\n\n".join([f"DOC:{c['doc_id']}#CHUNK:{c['chunk_index']}\n{c['text_snippet']}" for c in context_items])
    user_prompt = f"CONTEXT:\n{context_text}\n\nPERGUNTA: {question}\n\nResponda em Português, cite apenas os trechos acima usando doc_id e chunk_index. Se não souber, diga 'Não encontrado'."
    system_prompt = "Assistente jurídico. Cite as fontes (doc_id e chunk_index). Seja sucinto."
    answer = call_openai_chat(system_prompt, user_prompt)
    return {
        'answer': answer,
        'retrieved': context_items
    }


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action='store_true', help="Ingest PDFs e construir índice")
    parser.add_argument("--query", type=str, help="Fazer uma pergunta (RAG)")
    args = parser.parse_args()

    model = init_embedding_model()

    if args.ingest:
        ingest_all_pdfs(PDF_FOLDER, model)

    if args.query:
        out = query_rag(args.query, model)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

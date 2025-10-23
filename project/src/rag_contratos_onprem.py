"""
RAG Híbrido Offline (PDFs + JSON)

- Embeddings persistidos no disco, busca totalmente offline.
- Apenas os trechos mais relevantes são enviados para o OpenAI GPT, reduzindo custo e tokens.

Execução:

1️⃣ Ingestão de PDFs e JSON:
   python rag_contratos_onprem.py --ingest
   # Processa PDFs e JSON, gera embeddings e salva localmente (FAISS + JSON embeddings).

2️⃣ Consulta completa (resposta + trechos):
   python rag_contratos_onprem.py --query "PERGUNTA"

3️⃣ Consulta forçada (top JSON + top PDF, resposta + trechos):
   python rag_contratos_onprem.py --query-forced "PERGUNTA"

4️⃣ Consulta simples (apenas resposta do GPT):
   python rag_contratos_onprem.py --simple "PERGUNTA"

5️⃣ Consulta simples forçada (top JSON + PDF, apenas resposta):
   python rag_contratos_onprem.py --simple-forced "PERGUNTA"

Observações:

- Pipeline offline: PDFs e JSON processados e armazenados localmente.
- Top K e outros parâmetros podem ser ajustados no código.
"""


import os
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# -------------------- Config --------------------
load_dotenv()

PDF_FOLDER: Path = Path(os.getenv("PDF_FOLDER", "pdfs"))
DOCNAME_JSON_PATH: Path = Path(os.getenv("DOCNAME_JSON", ""))
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

INDEX_DIR: Path = Path("vector_index")
INDEX_FILE: Path = INDEX_DIR / "faiss.index"
METADATA_FILE: Path = INDEX_DIR / "metadata.jsonl"
JSON_EMB_FILE: Path = INDEX_DIR / "json_embeddings.json"

CHUNK_CHAR_SIZE: int = 3000
CHUNK_OVERLAP: int = 400
TOP_K: int = 6

# Config logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Inicializa cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------- Funções utilitárias --------------------
def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def extract_text_from_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    text = "\n".join(p.get_text("text") for p in doc)
    doc.close()
    return text


# -------------------- Embeddings --------------------
def get_openai_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    if not text.strip():
        return np.zeros(3072, dtype="float32")
    resp = client.embeddings.create(input=[text], model=model)
    return np.array(resp.data[0].embedding, dtype="float32")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


# -------------------- Chunking --------------------
def smart_chunk(text: str, chunk_size: int = CHUNK_CHAR_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    t = text.replace("\r", "")
    headings_pattern = re.compile(r"(?=\n\s*(CLÁUSULA|Cláusula|ARTIGO|Artigo|§))", re.IGNORECASE)
    parts = [s.strip() for s in headings_pattern.split(t) if s.strip()]

    if len(parts) >= 4:
        chunks, cur = [], ""
        for p in parts:
            if len(cur) + len(p) <= chunk_size:
                cur += "\n" + p
            else:
                chunks.append(cur.strip())
                cur = p
        if cur:
            chunks.append(cur.strip())
        return [(c, 0, 0) for c in chunks]

    # fallback
    out, i, L = [], 0, len(t)
    while i < L:
        end = min(i + chunk_size, L)
        out.append((t[i:end], i, end))
        i = max(end - overlap, end)
    return out


# -------------------- FAISS Index --------------------
def ensure_index_dir():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


class SimpleFAISSIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def save(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            for m in self.metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls):
        idx = faiss.read_index(str(INDEX_FILE))
        obj = cls(idx.d)
        obj.index = idx
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            obj.metadatas = [json.loads(line) for line in f]
        return obj

    def search(self, vector: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        D, I = self.index.search(vector, top_k)
        results = []
        for dist_list, idx_list in zip(D, I):
            for dist, idx in zip(dist_list, idx_list):
                if 0 <= idx < len(self.metadatas):
                    results.append({"score": float(dist), "metadata": self.metadatas[idx]})
        return results


# -------------------- Ingestão PDFs Offline --------------------
def ingest_all_pdfs():
    """Ingestão de PDFs com verificação de novos arquivos"""
    ensure_index_dir()
    metadata_items, vectors = [], []

    existing_index = None
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        existing_index = SimpleFAISSIndex.load()
        logging.info("FAISS index existente carregado.")
    else:
        logging.info("Criando novo FAISS index.")

    for pdf_path in tqdm(list(PDF_FOLDER.glob("*.pdf")), desc="Processando PDFs"):
        file_hash = sha1_of_file(pdf_path)
        # Verifica se já existe embedding para esse arquivo
        if existing_index and any(md["file_hash"] == file_hash for md in existing_index.metadatas):
            continue  # já processado

        doc_id = pdf_path.stem
        text = extract_text_from_pdf(pdf_path)
        chunks = smart_chunk(text)

        for i, (chunk_text, start, end) in enumerate(chunks):
            emb = get_openai_embedding(chunk_text)
            vectors.append(emb)
            metadata_items.append({
                "doc_id": doc_id,
                "file": pdf_path.name,
                "chunk_index": i,
                "text_snippet": chunk_text[:1500],
                "file_hash": file_hash
            })

    if not vectors:
        logging.info("Nenhum PDF novo encontrado.")
        return

    arr = np.vstack(vectors)
    faiss_idx = existing_index if existing_index else SimpleFAISSIndex(arr.shape[1])
    faiss_idx.add(arr, metadata_items)
    faiss_idx.save()
    logging.info("✅ FAISS index atualizado.")


# -------------------- Ingestão JSON Offline --------------------
def load_docname_json_embeddings() -> List[Dict[str, Any]]:
    """Carrega JSON embeddings, gera apenas novos"""
    ensure_index_dir()
    existing = {}
    if JSON_EMB_FILE.exists():
        with open(JSON_EMB_FILE, "r", encoding="utf-8") as f:
            existing = {e["key"]: e for e in json.load(f)}

    new_entries = []
    if not DOCNAME_JSON_PATH.exists():
        logging.warning("DOCNAME_JSON não encontrado.")
        return []

    with open(DOCNAME_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for file, people in data.items():
        for person in people:
            key = f"{file}-{person.get('nome','')}-{person.get('documento','')}"
            if key in existing:
                new_entries.append(existing[key])
            else:
                texto = f"{person.get('nome', '')} - {person.get('documento', '')} (arquivo: {file})"
                emb = get_openai_embedding(texto)
                entry = {"key": key, "name": texto, "embedding": emb.tolist(), "original": person}
                new_entries.append(entry)

    # Salva embeddings JSON atualizadas
    with open(JSON_EMB_FILE, "w", encoding="utf-8") as f:
        json.dump(new_entries, f, ensure_ascii=False, indent=2)

    # Converte embeddings para np.array
    for e in new_entries:
        e["embedding"] = np.array(e["embedding"], dtype="float32")

    return new_entries


# -------------------- Consulta Híbrida --------------------
def query_rag(question: str, json_embeddings: List[Dict[str, Any]], top_k: int = TOP_K) -> Dict[str, Any]:
    """Consulta RAG híbrido (PDF + JSON) totalmente offline para busca"""
    qv = get_openai_embedding(question)
    qv_norm = qv / np.linalg.norm(qv)

    # Busca JSON
    json_scores = [(cosine_similarity(qv_norm, e["embedding"]), e) for e in json_embeddings]
    json_scores.sort(reverse=True, key=lambda x: x[0])
    json_top = json_scores[:top_k]

    # Busca FAISS
    idx = SimpleFAISSIndex.load()
    hits = idx.search(np.expand_dims(qv, 0), top_k=top_k)

    # Combina resultados
    combined = []
    for score, e in json_top:
        combined.append({
            "source": "json",
            "score": float(score),
            "text_snippet": e["name"],
            "metadata": e["original"]
        })
    for h in hits:
        combined.append({
            "source": "pdf",
            "score": 1 / (1 + h["score"]),
            "text_snippet": h["metadata"]["text_snippet"],
            "metadata": h["metadata"]
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    combined_top = combined[:top_k]

    # OpenAI recebe apenas os trechos mais relevantes
    context_text = "\n\n---\n\n".join([f"[{c['source'].upper()}] {c['text_snippet']}" for c in combined_top])
    prompt = (
        f"Você é um assistente jurídico. Use os trechos abaixo (PDFs e JSON) para responder:\n\n"
        f"{context_text}\n\nPergunta: {question}\n\n"
        f"Responda em Português de forma objetiva, citando a origem (PDF ou JSON)."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistente jurídico RAG híbrido."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    answer = resp.choices[0].message.content
    return {"answer": answer, "retrieved": combined_top}


# -------------------- Consulta Híbrida Simplificada --------------------
def query_rag_simple(question: str, json_embeddings: List[Dict[str, Any]], top_k: int = TOP_K) -> str:
    """
    Consulta RAG híbrido (PDF + JSON) e retorna apenas a resposta do GPT.
    Mantém a busca offline, mas não retorna os trechos recuperados.
    """
    qv = get_openai_embedding(question)
    qv_norm = qv / np.linalg.norm(qv)

    # Busca JSON
    json_scores = [(cosine_similarity(qv_norm, e["embedding"]), e) for e in json_embeddings]
    json_scores.sort(reverse=True, key=lambda x: x[0])
    json_top = json_scores[:top_k]

    # Busca FAISS
    idx = SimpleFAISSIndex.load()
    hits = idx.search(np.expand_dims(qv, 0), top_k=top_k)

    # Combina resultados
    combined = []
    for score, e in json_top:
        combined.append({
            "source": "json",
            "score": float(score),
            "text_snippet": e["name"],
            "metadata": e["original"]
        })
    for h in hits:
        combined.append({
            "source": "pdf",
            "score": 1 / (1 + h["score"]),
            "text_snippet": h["metadata"]["text_snippet"],
            "metadata": h["metadata"]
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    combined_top = combined[:top_k]

    # OpenAI recebe apenas os trechos mais relevantes
    context_text = "\n\n---\n\n".join([f"[{c['source'].upper()}] {c['text_snippet']}" for c in combined_top])
    prompt = (
        f"Você é um assistente jurídico. Use os trechos abaixo (PDFs e JSON) para responder:\n\n"
        f"{context_text}\n\nPergunta: {question}\n\n"
        f"Responda em Português de forma objetiva, citando a origem (PDF ou JSON)."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistente jurídico RAG híbrido."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    return resp.choices[0].message.content

def query_rag_forced(question: str, json_embeddings, top_k_json=5, top_k_pdf=5):
    """
    Consulta RAG híbrido (PDF + JSON) garantindo que sempre pegue
    top_k_json itens do JSON e top_k_pdf chunks do PDF.
    """
    qv = get_openai_embedding(question)
    qv_norm = qv / np.linalg.norm(qv)

    # Busca JSON
    json_scores = [(cosine_similarity(qv_norm, e["embedding"]), e) for e in json_embeddings]
    json_scores.sort(reverse=True, key=lambda x: x[0])
    json_top = json_scores[:top_k_json]

    # Busca FAISS (PDF)
    idx = SimpleFAISSIndex.load()
    hits = idx.search(np.expand_dims(qv, 0), top_k=top_k_pdf)

    # Combina resultados forçando top-k de cada
    combined = []
    for score, e in json_top:
        combined.append({
            "source": "json",
            "score": float(score),
            "text_snippet": e["name"],
            "metadata": e["original"]
        })
    for h in hits:
        combined.append({
            "source": "pdf",
            "score": 1 / (1 + h["score"]),
            "text_snippet": h["metadata"]["text_snippet"],
            "metadata": h["metadata"]
        })

    # Ordenar combinado por score não é estritamente necessário aqui, mas pode manter
    combined_top = combined[:top_k_json + top_k_pdf]

    # Prompt para GPT
    context_text = "\n\n---\n\n".join([f"[{c['source'].upper()}] {c['text_snippet']}" for c in combined_top])
    prompt = (
        f"Você é um assistente jurídico. Use os trechos abaixo (PDFs e JSON) para responder:\n\n"
        f"{context_text}\n\nPergunta: {question}\n\n"
        f"Responda em Português de forma objetiva, citando a origem (PDF ou JSON)."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistente jurídico RAG híbrido."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    return {"answer": resp.choices[0].message.content, "retrieved": combined_top}

def query_rag_simple_forced(question: str, json_embeddings, top_k_json=5, top_k_pdf=5) -> str:
    """
    Consulta RAG híbrido (PDF + JSON) garantindo que sempre pegue top_k_json do JSON
    e top_k_pdf do PDF, mas retorna apenas a resposta do GPT.
    """
    # 1️⃣ Embedding da pergunta
    qv = get_openai_embedding(question)
    qv_norm = qv / np.linalg.norm(qv)

    # 2️⃣ Busca JSON
    json_scores = [(cosine_similarity(qv_norm, e["embedding"]), e) for e in json_embeddings]
    json_scores.sort(reverse=True, key=lambda x: x[0])
    json_top = json_scores[:top_k_json]

    # 3️⃣ Busca FAISS (PDF)
    idx = SimpleFAISSIndex.load()
    hits = idx.search(np.expand_dims(qv, 0), top_k=top_k_pdf)

    # 4️⃣ Combina resultados
    combined = []
    for score, e in json_top:
        combined.append({
            "source": "json",
            "score": float(score),
            "text_snippet": e["name"],
            "metadata": e["original"]
        })
    for h in hits:
        combined.append({
            "source": "pdf",
            "score": 1 / (1 + h["score"]),
            "text_snippet": h["metadata"]["text_snippet"],
            "metadata": h["metadata"]
        })

    # Mantemos todos os resultados forçados
    combined_top = combined[:top_k_json + top_k_pdf]

    # 5️⃣ Construção do prompt
    context_text = "\n\n---\n\n".join([f"[{c['source'].upper()}] {c['text_snippet']}" for c in combined_top])
    prompt = (
        f"Você é um assistente jurídico. Use os trechos abaixo (PDFs e JSON) para responder:\n\n"
        f"{context_text}\n\nPergunta: {question}\n\n"
        f"Responda em Português de forma objetiva, citando a origem (PDF ou JSON)."
    )

    # 6️⃣ Chamada GPT
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistente jurídico RAG híbrido."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    return resp.choices[0].message.content



# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="RAG híbrido offline (PDF + JSON)")
    parser.add_argument("--ingest", action="store_true", help="Ingestão de PDFs e JSON")
    parser.add_argument("--query", type=str, help="Consulta ao RAG (resposta + trechos recuperados)")
    parser.add_argument("--query-forced", type=str, help="Consulta híbrida forçando top JSON + PDF (resposta + trechos)")
    parser.add_argument("--simple", type=str, help="Consulta simplificada ao RAG (apenas resposta do GPT)")
    parser.add_argument("--simple-forced", type=str, help="Consulta simplificada forçando top JSON + PDF (apenas resposta do GPT)")

    # ✅ Aqui está a correção: parse args primeiro
    args = parser.parse_args()

    if args.ingest:
        ingest_all_pdfs()
        load_docname_json_embeddings()
        logging.info("✅ Ingestão concluída.")

    elif args.query:
        json_embeds = load_docname_json_embeddings()
        out = query_rag(args.query, json_embeds)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.query_forced:
        json_embeds = load_docname_json_embeddings()
        out = query_rag_forced(args.query_forced, json_embeds)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.simple:
        json_embeds = load_docname_json_embeddings()
        resposta = query_rag_simple(args.simple, json_embeds)
        print(resposta)

    elif args.simple_forced:
        json_embeds = load_docname_json_embeddings()
        resposta = query_rag_simple_forced(args.simple_forced, json_embeds)
        print(resposta)

    else:
        logging.info(
            "Use --ingest para ingestão, "
            "--query ou --query-forced para consulta completa, "
            "ou --simple / --simple-forced para resposta direta."
        )


if __name__ == "__main__":
    main()

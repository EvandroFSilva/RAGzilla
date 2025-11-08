from __future__ import annotations

"""Streamlit interface for the RAG Web Contracts system."""

import os
from typing import List

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Web Contracts", layout="wide")
st.title("RAG Web Contracts")
st.caption("Busque, selecione e consulte contratos armazenados no Google Drive.")


@st.cache_data(show_spinner=False)
def search_documents(query: str) -> List[dict]:
    response = requests.post(f"{API_URL}/search", json={"query": query}, timeout=30)
    response.raise_for_status()
    return response.json()


def ask_question(query: str, selected_docs: List[str]) -> dict:
    response = requests.post(
        f"{API_URL}/query",
        json={"query": query, "selected_docs": selected_docs},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# --------------------------
# Session state management
# --------------------------
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = set()  # document_id strings


with st.form("search_form"):
    user_query = st.text_input(
        "Pergunta inicial",
        value=st.session_state.search_query,
        placeholder="Busque por nomes, clausulas ou numeros de contrato...",
    )
    submitted = st.form_submit_button("Buscar documentos")

if submitted and user_query.strip():
    with st.spinner("Consultando base de documentos..."):
        try:
            results = search_documents(user_query.strip())
            st.session_state.search_query = user_query.strip()
            st.session_state.search_results = results
            # reset selection on new search
            st.session_state.selected_docs = set()
            # reset all checkbox keys for a clean UI
            for item in results:
                st.session_state[f"select_{item['document_id']}"] = False
        except requests.HTTPError as error:
            st.error(f"Erro ao buscar documentos: {error.response.text}")
        except Exception as error:  # pragma: no cover - interactive
            st.error(f"Falha inesperada na busca: {error}")

results: List[dict] = st.session_state.search_results

if results:
    st.subheader("Documentos encontrados")
    left, right = st.columns([1, 1])

    with left:
        st.caption("Resultados da busca")
        col_a, col_b = st.columns([1, 1])
        if col_a.button("Selecionar tudo"):
            ids = {item["document_id"] for item in results}
            st.session_state.selected_docs = set(ids)
            for doc_id in ids:
                st.session_state[f"select_{doc_id}"] = True
        if col_b.button("Limpar selecao"):
            for item in results:
                st.session_state[f"select_{item['document_id']}"] = False
            st.session_state.selected_docs = set()

        # Render checkboxes for each result
        for item in results:
            label = f"{item['document_name']} (score {item['score']:.3f})"
            doc_id = item["document_id"]
            key = f"select_{doc_id}"
            default_checked = doc_id in st.session_state.selected_docs
            checked = st.checkbox(label, value=st.session_state.get(key, default_checked), key=key)
            if checked:
                st.session_state.selected_docs.add(doc_id)
            else:
                st.session_state.selected_docs.discard(doc_id)

    with right:
        st.caption("Selecionados")
        if not st.session_state.selected_docs:
            st.info("Nenhum documento selecionado.")
        else:
            # Show selected document names in order of appearance
            for item in results:
                if item["document_id"] in st.session_state.selected_docs:
                    st.write(f"- {item['document_name']}")

selected_docs = sorted(list(st.session_state.selected_docs))

st.markdown("---")
st.header("Pergunte sobre os documentos selecionados")
question = st.text_area(
    "Pergunta",
    placeholder="Qual o prazo de validade dos contratos selecionados?",
)
if st.button("Gerar resposta", type="primary"):
    if not selected_docs:
        st.warning("Selecione ao menos um documento antes de perguntar.")
    elif not question.strip():
        st.warning("Informe a pergunta a ser respondida.")
    else:
        with st.spinner("Consultando modelos e gerando resposta..."):
            try:
                answer = ask_question(question.strip(), selected_docs)
            except requests.HTTPError as error:
                st.error(f"Erro ao gerar resposta: {error.response.text}")
            except Exception as error:  # pragma: no cover - interactive
                st.error(f"Falha inesperada: {error}")
            else:
                st.success("Resposta gerada com sucesso!")
                st.markdown(f"### Resposta\n{answer.get('answer', '')}")
                st.markdown("#### Fontes")
                for source in answer.get("sources", []):
                    # Backend provides 'document', 'page_number', and 'chunk' (text)
                    doc = source.get("document")
                    page = source.get("page_number")
                    chunk_text = source.get("chunk") or source.get("text")
                    with st.expander(f"{doc} (pagina {page})"):
                        st.write(chunk_text)


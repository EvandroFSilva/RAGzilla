from __future__ import annotations

"""Streamlit interface for the RAG Web Contracts system."""

import os
from typing import List

import requests
import streamlit as st
import base64

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


def fetch_pdf_bytes(document_id: str) -> bytes | None:
    """Baixa o PDF completo do backend para exibir no preview."""
    try:
        resp = requests.get(f"{API_URL}/document/{document_id}/file", timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


# --------------------------
# Session state management
# --------------------------
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = set()  # document_id strings
if "preview_doc" not in st.session_state:
    st.session_state.preview_doc = None  # documento selecionado para preview


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
            st.session_state.selected_docs = set()
            st.session_state.preview_doc = None  # reset preview

            # reset checkboxes
            for item in results:
                st.session_state[f"select_{item['document_id']}"] = False

        except requests.HTTPError as error:
            st.error(f"Erro ao buscar documentos: {error.response.text}")
        except Exception as error:
            st.error(f"Falha inesperada na busca: {error}")

results: List[dict] = st.session_state.search_results

# ===========================================
#           RESULTADOS + PREVIEW
# ===========================================
if results:
    st.subheader("Documentos encontrados")
    left, right = st.columns([1, 1])

    # -------------------------
    # COLUNA ESQUERDA (lista)
    # -------------------------
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
            st.session_state.preview_doc = None

        # Lista de resultados com checkbox
        for item in results:
            label = f"{item['document_name']} (score {item['score']:.3f})"
            doc_id = item["document_id"]
            key = f"select_{doc_id}"

            checked = st.checkbox(
                label,
                value=st.session_state.get(key, False),
                key=key,
            )

            if checked:
                st.session_state.selected_docs.add(doc_id)

                # Quando marcar, virar preview atual
                st.session_state.preview_doc = doc_id

            else:
                st.session_state.selected_docs.discard(doc_id)

                # Se desmarcou o documento que estava no preview, limpa preview
                if st.session_state.preview_doc == doc_id:
                    st.session_state.preview_doc = None

    # -------------------------
    # COLUNA DIREITA (PREVIEW)
    # -------------------------
    with right:
        st.caption("Visualização do documento selecionado")

        if st.session_state.preview_doc is None:
            st.info("Selecione um documento para visualizar o PDF.")
        else:
            document_id = st.session_state.preview_doc

            st.write(f"**Documento:** {document_id}")

            pdf_data = fetch_pdf_bytes(document_id)

            if pdf_data is None:
                st.error("Falha ao carregar o PDF.")
            else:
                # Exibir PDF em iframe
                b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
                pdf_display = f"""
                    <iframe 
                        src="data:application/pdf;base64,{b64_pdf}"
                        width="100%" height="700px"
                        type="application/pdf">
                    </iframe>
                """
                st.markdown(pdf_display, unsafe_allow_html=True)

                # Botão para baixar PDF
                st.download_button(
                    "Baixar PDF",
                    data=pdf_data,
                    file_name=document_id,
                    mime="application/pdf",
                )

# ===========================================
#          PERGUNTA AO MODELO
# ===========================================
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
            except Exception as error:
                st.error(f"Falha inesperada: {error}")
            else:
                st.success("Resposta gerada com sucesso!")
                st.markdown(f"### Resposta\n{answer.get('answer', '')}")
                st.markdown("#### Fontes")
                for source in answer.get("sources", []):
                    doc = source.get("document")
                    page = source.get("page_number")
                    chunk_text = source.get("chunk") or source.get("text")
                    with st.expander(f"{doc} (pagina {page})"):
                        st.write(chunk_text)

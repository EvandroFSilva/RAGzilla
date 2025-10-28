"""Aplicação Streamlit para recuperação e geração de respostas com Ollama offline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import streamlit as st

from .rag_ollama_pipeline import (
    DOCUMENTS_DIR,
    answer_question,
    ingest_documents,
    indexes_exist,
    retrieve_context,
    search_documents,
)


LOGGER = logging.getLogger(__name__)


def _init_session_state() -> None:
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = {}  # type: Dict[str, str]
    if "search_results" not in st.session_state:
        st.session_state.search_results = []


def _render_header(documents_dir: Path) -> None:
    st.title("RAG Offline com Ollama")
    st.caption(
        "Faça buscas por documentos com embeddings locais (embeddinggemma) e "
        "gere respostas com o modelo Qwen3-8B via Ollama."
    )
    st.write(f"Diretório de documentos: `{documents_dir}`")


def _ensure_indexes_ui(documents_dir: Path) -> bool:
    if indexes_exist():
        return True

    st.warning(
        "Os índices vetoriais ainda não foram gerados. Clique no botão abaixo para processar "
        "os documentos disponíveis."
    )
    if st.button("Gerar índices", type="primary"):
        with st.spinner("Gerando embeddings e índices vetoriais..."):
            summary = ingest_documents(documents_dir)
        st.success(
            f"Indexação concluída: {summary['documents']} documentos, "
            f"{summary['discovery_chunks']} chunks de descoberta e "
            f"{summary['context_chunks']} chunks de contexto."
        )
        return True

    return False


def _render_search_area() -> None:
    st.subheader("1. Buscar documentos")
    query = st.text_input(
        "Digite uma pergunta ou termo para localizar documentos",
        placeholder="Ex.: Qual contrato menciona João Silva?",
        key="discovery_query",
    )

    if st.button("Buscar documentos", use_container_width=True):
        if not query.strip():
            st.warning("Informe uma busca antes de continuar.")
        else:
            with st.spinner("Consultando índice de descoberta..."):
                st.session_state.search_results = search_documents(query, top_k=10)
            if not st.session_state.search_results:
                st.info("Nenhum trecho relevante encontrado para a busca realizada.")

    if st.session_state.search_results:
        st.write("### Resultados")
        for idx, result in enumerate(st.session_state.search_results, start=1):
            doc_id = result["doc_id"]
            doc_name = result["document_name"]
            with st.container():
                st.markdown(
                    f"**{idx}. {doc_name}** — doc_id: `{doc_id}` — score: {result['score']:.3f}"
                )
                st.markdown(f"> {result['snippet']}")
                cols = st.columns([1, 1])
                if cols[0].button(
                    "Selecionar documento",
                    key=f"select_{doc_id}_{idx}",
                    disabled=doc_id in st.session_state.selected_docs,
                ):
                    st.session_state.selected_docs[doc_id] = doc_name
                    st.experimental_rerun()
                cols[1].write(" ")  # espaçamento


def _render_selected_documents() -> None:
    st.subheader("2. Documentos selecionados")
    selected = st.session_state.selected_docs
    if not selected:
        st.info("Nenhum documento selecionado até o momento.")
    else:
        for doc_id, doc_name in selected.items():
            st.markdown(f"- `{doc_id}` — {doc_name}")

    cols = st.columns([1, 1])
    if cols[0].button("Limpar seleção", use_container_width=True):
        st.session_state.selected_docs.clear()
        st.experimental_rerun()

    if cols[1].button("Ver chunks relevantes", use_container_width=True, disabled=not selected):
        with st.spinner("Buscando trechos nos documentos selecionados..."):
            previews = retrieve_context(
                "Resumo dos documentos selecionados", selected.keys(), top_k=5
            )
        if not previews:
            st.info("Nenhum trecho encontrado para os documentos selecionados.")
        else:
            for preview in previews:
                st.markdown(
                    f"**Documento:** {preview['document_name']} — chunk {preview['chunk_index']}"
                )
                st.markdown(f"> {preview['text'][:400]}...")


def _render_question_area() -> None:
    st.subheader("3. Fazer pergunta sobre os documentos selecionados")
    question = st.text_area(
        "Escreva a pergunta",
        placeholder="Ex.: Quais cláusulas falam sobre reajuste anual?",
        key="qa_question",
    )

    if st.button("Gerar resposta", type="primary", use_container_width=True):
        if not st.session_state.selected_docs:
            st.warning("Selecione ao menos um documento antes de perguntar.")
            return
        if not question.strip():
            st.warning("Informe uma pergunta válida.")
            return

        with st.spinner("Consultando documentos selecionados e gerando resposta..."):
            result = answer_question(question, st.session_state.selected_docs.keys(), top_k=6)

        st.success("Resposta gerada!")
        st.markdown("### Resposta")
        st.write(result["answer"])

        if result["contexts"]:
            with st.expander("Trechos utilizados na resposta"):
                for ctx in result["contexts"]:
                    st.markdown(
                        f"**Documento:** {ctx['document_name']} — chunk {ctx['chunk_index']}"
                    )
                    st.markdown(f"> {ctx['text']}")


def main() -> None:  # pragma: no cover - função para execução Streamlit
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    documents_dir = DOCUMENTS_DIR

    _init_session_state()
    _render_header(documents_dir)

    if not _ensure_indexes_ui(documents_dir):
        return

    _render_search_area()
    st.divider()
    _render_selected_documents()
    st.divider()
    _render_question_area()


if __name__ == "__main__":  # pragma: no cover
    main()

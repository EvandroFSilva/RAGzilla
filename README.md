# RAGzilla
 Plataforma inteligente baseada em RAG que permite consultas em linguagem natural sobre documentos jurídicos. Utilizando técnicas avançadas de IA, o sistema recupera trechos relevantes de leis, contratos, sentenças e pareceres, e gera respostas precisas e contextualizadas, facilitando o acesso à informação jurídica com agilidade e confiança.
 https://drive.google.com/drive/folders/1iIz7fdd8bNsiaIPrEa7SucWGaNptVyaC?usp=sharing

## RAG offline com Ollama

O repositório agora inclui uma aplicação **totalmente offline** que utiliza os modelos
`embeddinggemma` (embeddings) e `Qwen3-8B` (LLM) disponibilizados pelo [Ollama](https://ollama.com/).
Ela permite:

- Procurar documentos por meio de chunks curtos (~50 tokens) para localizar nomes e números.
- Selecionar vários documentos relevantes e limpar a seleção sempre que necessário.
- Formular perguntas considerando apenas os documentos escolhidos, utilizando chunks maiores (~500 tokens) como contexto.

### Pré-requisitos

- Servidor Ollama em execução com os modelos `embeddinggemma` e `Qwen3-8B` previamente baixados.
- Documentos `.txt`, `.md` ou `.pdf` armazenados em `project/data/documents` (ou outro diretório definido na variável `RAG_DOCUMENTS_DIR`).

### Preparação dos índices

```bash
python -m project.src.rag_ollama_pipeline --ingest
```

O comando acima lê todos os documentos suportados, gera os embeddings com chunks pequenos e grandes e salva os índices na pasta `project/data/indexes`.

### Execução da aplicação

```bash
streamlit run project/src/rag_ollama_app.py
```

1. Realize buscas para encontrar os documentos de interesse.
2. Selecione um ou mais documentos usando os botões exibidos.
3. Faça perguntas sobre os documentos selecionados e visualize os trechos utilizados na resposta.

Também é possível interagir via linha de comando:

```bash
# Busca rápida
python -m project.src.rag_ollama_pipeline --search "nome do colaborador"

# Pergunta com lista de doc_ids selecionados
python -m project.src.rag_ollama_pipeline --question "Qual é a vigência do contrato?" --selected contrato_a contrato_b
```

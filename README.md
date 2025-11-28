# RAG Web Contracts

Sistema completo de Recuperação Aumentada por Geração (RAG) que permite buscar, selecionar e consultar contratos armazenados no Google Drive por meio de uma API FastAPI e interface Streamlit. A solução utiliza embeddings da OpenAI, índices FAISS locais e suporta OCR automático para PDFs digitalizados.

## Visão geral da arquitetura

```
Google Drive --> Ingestão (OCR + chunking curto/long) --> Embeddings OpenAI --> FAISS (curto/long) --> API FastAPI --> Streamlit
```

### Componentes principais

| Pasta/arquivo | Descrição |
| ------------- | --------- |
| `backend/ingestion.py` | Pipeline de ingestão com sincronização do Google Drive, OCR e geração de índices FAISS curtos (20 tokens) e longos (500 tokens). |
| `backend/retrieval.py` | Consulta aos índices FAISS (busca inicial curta + contexto longo para resposta). |
| `backend/query.py` | Geração de respostas com o modelo `gpt-4o-mini`, garantindo citações das fontes utilizadas. |
| `backend/main.py` | API FastAPI com endpoints `/health`, `/search` e `/query`. |
| `frontend/app.py` | Interface Streamlit para busca, seleção de documentos e perguntas sobre os contratos escolhidos. |
| `backend/drive_utils.py` | Download e verificação de PDFs no Google Drive utilizando credenciais de serviço. |
| `backend/settings.py` | Configuração centralizada via variáveis de ambiente. |
| `requirements.txt` | Dependências Python necessárias. |
| `Dockerfile` / `docker-compose.yml` | Execução containerizada da API e da interface web. |

## Pré-requisitos

- Python 3.11+
- Credenciais de serviço do Google Drive com acesso somente leitura à pasta configurada
- Variável `OPENAI_API_KEY` com chave de API válida
- Biblioteca Tesseract instalada no sistema (ex.: `apt-get install tesseract-ocr`)

## Configuração

1. Crie e configure um arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sk-...
GOOGLE_DRIVE_FOLDER_ID=...
GOOGLE_APPLICATION_CREDENTIALS=/caminho/para/service_account.json
```

2. Instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Execute a ingestão dos documentos (veja as variações do comando abaixo):

```bash
python backend/ingestion.py --sync-drive --recreate
ou
python -m backend.ingestion --sync-drive --recreate
```

### Variações do comando

- `python backend/ingestion.py` e `python -m backend.ingestion` executam exatamente a mesma rotina; a segunda forma usa o módulo como entrypoint (útil quando o diretório não está no `PYTHONPATH`).
- `--sync-drive` sincroniza a pasta configurada do Google Drive antes de qualquer processamento. Sem essa flag, apenas os PDFs que já estiverem em `data/raw` (ou pasta configurada) serão avaliados.
- `--recreate` força a reconstrução completa dos índices FAISS e dos arquivos `.npz`, ignorando o cache local de embeddings e chunks. Quando omitida, a pipeline roda de forma incremental: apenas PDFs novos ou alterados geram chunks/embeddings e as chamadas ao OpenAI são limitadas a esses arquivos, enquanto os demais são reutilizados diretamente do armazenamento local.

Independentemente da combinação escolhida, o comando irá executar OCR quando necessário, criar os chunks curtos/longos, gerar embeddings com o modelo `text-embedding-3-small` (ou somente para os arquivos novos/alterados) e salvar/atualizar os índices FAISS em `data/faiss_index_short/index.faiss` e `data/faiss_index_long/index.faiss`.

## Executando a API

```bash
uvicorn backend.main:app --reload
```

Endpoints principais:

- `POST /search` – recebe `{ "query": "..." }` e retorna a lista de documentos relevantes (via embeddings curtos).
- `POST /query` – recebe `{ "query": "...", "selected_docs": ["docA.pdf"] }`, busca os 5 chunks longos mais relevantes e envia ao `gpt-4o-mini` para gerar a resposta contextualizada.

## Interface Streamlit

Com a API em execução, em outro terminal rode:

```bash
streamlit run frontend/app.py
```

Fluxo sugerido:

1. Informe uma pergunta inicial para localizar documentos.
2. Selecione um ou mais arquivos apresentados.
3. Faça uma pergunta final. Apenas os 5 melhores trechos longos serão enviados ao modelo para gerar a resposta.
4. A resposta exibirá as fontes (trecho e nome do arquivo) utilizadas.

## Execução com Docker

O repositório inclui arquivos de containerização para simplificar o deploy local:

```bash
docker compose up --build
```

O serviço expõe:

- API FastAPI em `http://localhost:8000`
- Interface Streamlit em `http://localhost:8501`

Certifique-se de montar o diretório `data/` como volume para persistência dos índices FAISS.

## Testes

Rode os testes automatizados com:

```bash
pytest
```

Os testes cobrem utilitários de chunking e formatação de prompts.

## Logs

O backend utiliza logging estruturado via `logging.config.dictConfig`. Mensagens são emitidas para o stdout com timestamp, nível e módulo de origem.

## Licença

Distribuído apenas para fins educacionais.

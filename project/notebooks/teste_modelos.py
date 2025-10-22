import os
import json
import requests
import time
from PyPDF2 import PdfReader

# === Função para extrair texto da primeira página do PDF ===
def extrair_texto_pdf(caminho_pdf):
    try:
        reader = PdfReader(caminho_pdf)
        if reader.pages:
            return reader.pages[0].extract_text().strip()
        return ''
    except Exception as e:
        print(f"Erro ao ler PDF {caminho_pdf}: {e}")
        return ''

# === Gera o prompt para extração JSON padronizada ===
def gerar_prompt_json(texto):
    return f"""
**Verifique mais de uma vez para ter certeza que não esqueceu nenhum nome**.
**Retorne somente o JSON, sem texto antes ou depois**
Tarefa:
- Para cada nome, identifique os **documentos** associados (CPF, CNPJ, RG, número de inscrição etc.).
- Se o mesmo documento é associado a mais de um nome ou empresa, inclua todos.
- Use **JSON compacto**, sem quebras de linha dentro de strings.
- Procure em todo documento nomes de pessoas, empresas, sócios, lados ou representados e retorne **EXCLUSIVAMENTE** um JSON válido, no formato abaixo:
{{
  "envolvidos": [
    {{
      "nome": "pessoa, empresa, sócio, lado ou representado",
      "documento": "numero do documento"
    }},
    ...
  ]
}}

Contrato:
\"\"\"{texto}\"\"\"
"""

# === Chama o modelo local via Ollama (com retry automático) ===
def extrair_envolvidos_ollama(modelo, texto, max_retentativas=3, delay=10):
    prompt = gerar_prompt_json(texto)

    for tentativa in range(1, max_retentativas + 1):
        try:
            print(f"🌀 Tentativa {tentativa}/{max_retentativas} para {modelo}...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": modelo,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024
                    }
                },
                timeout=180
            )

            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                print(f"\n🧩 [{modelo}] RAW OUTPUT:\n{raw}\n")

                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    print(f"⚠️ Erro ao decodificar JSON para {modelo}")
                    return {"erro_json": raw}
            else:
                print(f"❌ Erro API {modelo} ({response.status_code}): {response.text}")

        except Exception as e:
            print(f"❌ Falha ao acessar {modelo}: {e}")

        # Aguarda antes de tentar novamente (se não for a última tentativa)
        if tentativa < max_retentativas:
            print(f"⏳ Aguardando {delay}s antes da nova tentativa...")
            time.sleep(delay)

    print(f"🚫 Todas as {max_retentativas} tentativas falharam para {modelo}.")
    return {"erro": f"Falha após {max_retentativas} tentativas."}

# === Processa PDFs e gera um arquivo por modelo ===
def processar_modelos(pasta_pdfs, modelos):
    for modelo in modelos:
        resultados_modelo = {}
        print(f"\n==============================")
        print(f"🤖 Iniciando testes com modelo: {modelo}")
        print(f"==============================")

        for arquivo in os.listdir(pasta_pdfs):
            if not arquivo.lower().endswith(".pdf"):
                continue

            caminho_pdf = os.path.join(pasta_pdfs, arquivo)
            print(f"📄 Lendo: {arquivo}")
            texto = extrair_texto_pdf(caminho_pdf)

            if not texto:
                print("⚠️ Nenhum texto encontrado.")
                continue

            resultado = extrair_envolvidos_ollama(modelo, texto)
            resultados_modelo[arquivo] = resultado

        # === Nome do arquivo de saída específico do modelo ===
        nome_arquivo = modelo.replace(":", "_").replace("-", "_") + ".json"
        caminho_saida = os.path.join("resultados_modelos", nome_arquivo)

        os.makedirs("resultados_modelos", exist_ok=True)
        with open(caminho_saida, "w", encoding="utf-8") as f:
            json.dump(resultados_modelo, f, indent=2, ensure_ascii=False)

        print(f"✅ Resultados do modelo '{modelo}' salvos em: {caminho_saida}")

# === Execução principal ===
if __name__ == "__main__":
    pasta = "documentos"

    modelos_teste = [
        "mistral:7b",
        "llama3:8b",
        "qwen2:7b",
        "phi3:mini",
        "codellama:7b-instruct",
        "deepseek-r1:8b",
        "deepseek-r1:7b"
        "qwen3:8b"
    ]

    print("🚀 Iniciando testes com múltiplos modelos Ollama...")
    processar_modelos(pasta, modelos_teste)

    print("\n🏁 Todos os modelos foram processados com sucesso!")

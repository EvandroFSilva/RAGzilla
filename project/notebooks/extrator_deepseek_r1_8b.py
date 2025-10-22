import os
import json
import requests
from PyPDF2 import PdfReader
import re
import fitz  # PyMuPDF
import re

# === Função para extrair texto de um PDF (robusta e limpa) ===
def extrair_texto_pdf(caminho_pdf):
    try:
        texto = ""
        with fitz.open(caminho_pdf) as pdf:
            if pdf.page_count > 0:
                page = pdf[0]  # primeira página
                texto = page.get_text("text") or ""

        # --- LIMPEZA ---
        texto = texto.replace('\n', ' ')  # quebra de linha → espaço
        texto = re.sub(r'\s*-\s*', '-', texto)      # "123 - 45" → "123-45"
        texto = re.sub(r'\s*\.\s*', '.', texto)     # "123 . 456" → "123.456"
        texto = re.sub(r'[^\x20-\x7EÀ-ÿ]', ' ', texto)  # remove símbolos ilegíveis
        texto = re.sub(r'\s{2,}', ' ', texto)       # múltiplos espaços → 1

        texto = texto.strip()
        return texto

    except Exception as e:
        print(f"❌ Erro ao ler PDF {caminho_pdf}: {e}")
        return ""

# === Prompt few-shot para retorno em JSON ===
def gerar_prompt(texto):
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

# === Função que envia texto ao modelo via Ollama com retry e robustez ===
def extrair_envolvidos(texto, tentativas=3):
    for tentativa in range(1, tentativas + 1):
        prompt = gerar_prompt(texto)  # REMOVIDO 'reforco'
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:8b",
                    "prompt": prompt,
                    "stream": False,                    
                    "options": {
                        "temperature": 0.3,                
                        "num_predict": 1024
                    }
                },
                timeout=180
            )

            if response.status_code != 200:
                print(f"❌ Erro na API ({response.status_code}): {response.text}")
                continue

            raw = (response.json().get("response") or "").strip()

            if not raw:
                print(f"⚠️ Resposta vazia na tentativa {tentativa}.")
                print(f"🔄 Tentando novamente... ({tentativa}/{tentativas})")
                continue

            print(f"\n🧩 RAW MODEL OUTPUT (tentativa {tentativa}):\n{raw}\n")

            json_data = limpar_json(raw)
            if json_data and isinstance(json_data, dict):
                return json_data.get("envolvidos", [])

            print(f"⚠️ Nenhum JSON válido encontrado na tentativa {tentativa}.")
        except Exception as e:
            print(f"⚠️ Erro ao acessar o modelo na tentativa {tentativa}: {e}")

        print(f"🔄 Tentando novamente... ({tentativa}/{tentativas})")

    print("❌ Falha após 3 tentativas. Retornando lista vazia.")
    return []


# === Sanitização e correção de JSON bruto ===
def limpar_json(raw):
    raw = re.sub(r'//.*', '', raw)
    raw = re.sub(r'#.*', '', raw)
    raw = raw.replace('```json', '').replace('```', '').strip()

    # Extrai bloco de JSON (objeto OU lista)
    match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
        # Se for lista, empacota no formato padrão
        if isinstance(data, list):
            return {"envolvidos": data}
        return data
    except json.JSONDecodeError:
        raw_corrigido = re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            data = json.loads(raw_corrigido)
            if isinstance(data, list):
                return {"envolvidos": data}
            return data
        except Exception as e:
            print(f"⚠️ Falha ao decodificar JSON mesmo após correção: {e}")
            return None



# === Função principal para processar todos os PDFs ===
def processar_pasta_pdfs(pasta):
    resultados = {}

    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith('.pdf'):
            caminho = os.path.join(pasta, arquivo)
            print(f"\n📄 Processando: {arquivo}")
            texto = extrair_texto_pdf(caminho)

            if not texto:
                print("⚠️ Nenhum texto extraído.")
                continue

            envolvidos = extrair_envolvidos(texto)
            resultados[arquivo] = envolvidos

    return resultados


# === Execução principal ===
# === Execução principal ===
if __name__ == "__main__":
    pasta_pdfs = r"project\data\raw"
    arquivo_saida = r"project\data\processed\deepseek_r1_8b.json"

    print("🚀 Iniciando extração de envolvidos...")
    resultado_final = processar_pasta_pdfs(pasta_pdfs)

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extração finalizada! Resultado salvo em: {arquivo_saida}")


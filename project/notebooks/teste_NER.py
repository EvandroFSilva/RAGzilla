import os
import re
import pdfplumber
import spacy
from dotenv import load_dotenv

# ==============================================
# 1. Configuração de pasta
# ==============================================
load_dotenv()
pdf_folder = os.getenv("PDF_FOLDER")

# ==============================================
# 2. Funções auxiliares
# ==============================================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'__', '', text)
    return text.strip()

# Stoplist para remover entidades que não são nomes/empresas reais
STOPWORDS_BOILERPLATE = {
    "parágrafo","paragrafo","cláusula","clausula","parte","partes",
    "objeto","contrato","testemunhas","artigo","pagina","página",
    "prazo","preço","preco","lei","lgpd","constituição","constituicao",
    "residentes","residente","extensão","extensao","recebimento",
    "condominio","condomínio","visibilidade","marketing","nps"
}

def _normalize_entity_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)  # normaliza espaços
    s = re.sub(r'^[^A-Za-zÀ-ÖØ-öø-ÿ]+|[^A-Za-zÀ-ÖØ-öø-ÿ]+$', '', s)
    return s

# ==============================================
# 3. Carregar spaCy
# ==============================================
# Modelo português — instale se necessário:
# python -m spacy download pt_core_news_sm
nlp = spacy.load("pt_core_news_sm")

def extract_spacy_entities(text: str):
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        label = ent.label_.upper()
        if label not in ("PER", "PERSON", "ORG"):
            continue

        ent_text = _normalize_entity_text(ent.text)
        if not ent_text:
            continue

        # --- Regras anti-ruído ---
        if re.search(r'\d', ent_text):  # elimina se tiver números
            continue
        if ent_text.isupper():  # elimina se tudo maiúsculo (ex: "PRAZO")
            continue
        if any(sw in ent_text.lower() for sw in STOPWORDS_BOILERPLATE):
            continue
        if len(ent_text) < 3:
            continue
        # exige pelo menos 2 palavras com letra maiúscula (para PER)
        if label in ("PER", "PERSON"):
            tokens = ent_text.split()
            caps = [w for w in tokens if w and w[0].isupper()]
            if len(caps) < 2:
                continue

        if ent_text not in seen:
            seen.add(ent_text)
            entities.append((ent_text, label))
    return entities


# ==============================================
# 4. Rodar NER nos PDFs
# ==============================================
for pdf_file in os.listdir(pdf_folder):
    if not pdf_file.lower().endswith(".pdf"):
        continue
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"\n📄 {pdf_file}")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = clean_text(page.extract_text())
            full_text += " " + page_text
    ents = extract_spacy_entities(full_text)
    if ents:
        for ent, label in ents:
            print(f"  - {ent} ({label})")
    else:
        print("  ⚠ Nenhuma entidade relevante encontrada")

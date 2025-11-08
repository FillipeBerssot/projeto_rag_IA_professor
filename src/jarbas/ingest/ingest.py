"""
ingest.py
=========
Pipeline de **ingestão** dos seus arquivos-fonte para gerar um corpus unificado
em JSONL com chunks de texto.

O que faz
---------
1) Caminha recursivamente por diretórios de origem (ex.: `data/sources`, `data`),
   filtrando extensões de interesse (.pdf, .md, .html/.htm, .txt, .rst) e
   ignorando diretórios ruído (ex.: .git, node_modules, _build etc.).
2) Extrai **texto puro** de cada arquivo suportado:
   - PDF via pypdf
   - Markdown via markdown-it → HTML → BeautifulSoup (texto)
   - HTML/HTM via BeautifulSoup, removendo tags de navegação
   - TXT/RST leitura direta
3) Limpa o texto (`clean_text`) e quebra em **chunks** (`chunk_text`).
4) Salva em `data/processed/corpus.jsonl` (um JSON por linha) com campos:
   - id: caminho + "::chunk{n}"
   - source: caminho relativo
   - chunk_id: índice do chunk no arquivo
   - text: conteúdo do chunk

Parâmetros globais
------------------
- EXTS: extensões textuais suportadas.
- MAX_FILE_MB: tamanho máximo (MB) por arquivo; acima disso será ignorado,
  exceto se o caminho estiver em `WHITELIST_LARGE`.
- WHITELIST_LARGE: arquivos grandes permitidos explicitamente.
- BINARY_EXTS: extensões binárias/imagens/arquivos compactados a ignorar.

Uso
---
    python -m src.ingest

Saída
-----
- Arquivo `data/processed/corpus.jsonl` contendo todos os chunks de texto.
"""

import os, json

from typing import Dict, List
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pypdf import PdfReader
from src.jarbas.utils.text import clean_text, chunk_text
from tqdm import tqdm


md = MarkdownIt()

EXTS = (".pdf", ".md", ".html", ".htm", ".txt", ".rst")
MAX_FILE_MB = 800
WHITELIST_LARGE = {
    "data/sources/dados matplotlib/matplotlib/Matplotlib.pdf",
    "data/sources/dados SQL/sql/sqlserver/sql-sql-server-ver17.pdf",
}

BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
               ".ico", ".ttf", ".woff", ".woff2", ".eot", ".mp4",
               ".mp3", ".zip", ".tar", ".gz", ".bz2"
            }

def read_pdf(path: str) -> str:
    """
    Extrai texto de um PDF página a página.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo PDF.

    Retorna
    -------
    str
        Texto concatenado das páginas do PDF. Retorna string vazia em caso de falha.

    Observações
    -----------
    - Usa `pypdf.PdfReader.extract_text()`.
    - Erros são capturados e logados como aviso, sem interromper a ingestão.
    """
        
    try:
        reader = PdfReader(path)
        texts = []

        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)

        return "\n".join(texts)
    
    except Exception as e:
        print(f"[WARN] PDF Falhou: {path} ({e})")
        return ""
    
def read_markdown(path: str) -> str:
    """
    Converte Markdown para texto plano.

    Estratégia
    ----------
    1) Lê o arquivo .md como string.
    2) Converte para HTML via `markdown_it`.
    3) Faz o parsing com BeautifulSoup e extrai texto.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo Markdown.

    Retorna
    -------
    str
        Texto extraído (ou string vazia em caso de erro).
    """
        
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            md_text = f.read()
        # converte MD -> HTML -> texto plano
        html = md.render(md_text)
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text("\n")
    
    except Exception as e:
        print(f"[WARN] MD falhou: {path} ({e})")
        return ""

def read_html(path: str) -> str:
    """
    Extrai texto de um arquivo HTML/HTM, removendo elementos de navegação.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo HTML/HTM.

    Retorna
    -------
    str
        Título (quando houver) + corpo como texto plano. String vazia em caso de erro.

    Notas
    -----
    - Remove tags típicas de navegação/estilo: script, style, nav, header, footer, aside, noscript.
    - Usa `html.parser` por padrão.
    """
        
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        # remove navegação/complements comuns
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()
        # puxa titulo quando existir
        title = soup.title.get_text(strip=True) if soup.title else ""
        body = soup.get_text("\n")

        return (title + "\n\n" + body).strip() if title else body
    except Exception as e:
        print(f"[WARN] HTML falhou: {path} ({e})")
        return ""

def read_textlike(path: str) -> str:
    """
    Lê arquivos de texto “puros” (ex.: .txt, .rst) diretamente.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo.

    Retorna
    -------
    str
        Conteúdo integral do arquivo como string. Vazio se falhar.
    """
    
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
        
    except Exception as e:
        print(f"[WARN] TXT/RST falhou: {path} ({e})")
        return ""

def extract_text(path: str) -> str:
    """
    Router de extração de texto conforme a extensão do arquivo.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo.

    Retorna
    -------
    str
        Texto extraído por uma das funções específicas:
        - PDF → `read_pdf`
        - MD → `read_markdown`
        - HTML/HTM → `read_html`
        - Demais (txt/rst) → `read_textlike`
    """
    
    p = path.lower()
    if p.endswith(".pdf"):
        return read_pdf(path)
    if p.endswith(".md"):
        return read_markdown(path)
    if p.endswith(".html") or p.endswith(".htm"):
        return read_html(path)
    # .txt / .rst / outros texto puro
    return read_textlike(path)

def walk_sources(root_dirs: List[str]) -> List[Dict]:
    """
    Caminha recursivamente pelas pastas de origem, extrai e chunk-a textos.

    Parâmetros
    ----------
    root_dirs : list[str]
        Lista de diretórios-raiz onde buscar arquivos (ex.: ["data/sources", "data"]).

    Retorna
    -------
    list[dict]
        Lista de registros (um por chunk) no formato:
        {
            "id": f"{path}::chunk{i}",
            "source": <caminho_relativo>,
            "chunk_id": i,
            "text": <texto_do_chunk>
        }

    Regras de varredura
    -------------------
    - Ignora diretórios ruidosos (ex.: .git, __pycache__, node_modules, _build...).
    - Ignora extensões binárias (BINARY_EXTS).
    - Aceita apenas extensões em `EXTS`.
    - Aplica limite de tamanho por arquivo (MAX_FILE_MB), com whitelist.

    Observações
    -----------
    - Usa barra de progresso (`tqdm`) no processamento dos arquivos.
    - `clean_text` normaliza espaços/linhas.
    - `chunk_text` segmenta o conteúdo em blocos para melhor recuperação posterior.
    """
    
    IGNORE_DIRS = {
        ".git", "__pycache__", "_build", "_static", "_sources",
        ".venv", "node_modules", "site", "build", "dist",
        "processed", "index" 
    }

    docs = []
    root_abs = os.path.abspath(os.getcwd())
    files_to_process = set()

    for root in root_dirs:
        for cur_dir, subdirs, files in os.walk(root):
            # filtra subdirs in-place (não desce em diretórios ignorados)
            subdirs[:] = [
                d for d in subdirs
                if d not in IGNORE_DIRS and not d.startswith(".")
            ]
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()

                if ext in BINARY_EXTS:
                    continue
                if not any(ext.endswith(e) for e in EXTS):
                    continue
                
                path = os.path.join(cur_dir, fn)

                try:
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                except OSError:
                    continue

                if size_mb > MAX_FILE_MB and path not in WHITELIST_LARGE:
                    print(f"[SKIP] Muito grande ({size_mb:.1f} MB): {path}")
                    continue

                files_to_process.add(path)

    # processamento com barra de progresso
    for path in tqdm(files_to_process, desc="Ingestão"):
        txt = extract_text(path)
        txt = clean_text(txt)
        
        if not txt:
            continue

        rel = os.path.relpath(path, start=root_abs)
        for i, ch in enumerate(chunk_text(txt)):
            docs.append({
                "id": f"{path}::chunk{i}",
                "source": rel,
                "chunk_id": i,
                "text": ch
            })

    return docs

if __name__ == "__main__":

    SOURCE_DIRS = ["data/sources", "data"]

    os.makedirs("data/processed", exist_ok=True)
    docs = walk_sources(SOURCE_DIRS)
    out_path = "data/processed/corpus.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"OK: {len(docs)} chunks salvos em {out_path}")

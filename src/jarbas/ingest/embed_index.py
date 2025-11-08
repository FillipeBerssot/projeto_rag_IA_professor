"""
embed_index.py
================
Gera o índice vetorial FAISS a partir do corpus processado (JSONL).

O que faz
---------
1) Lê `data/processed/corpus.jsonl`, onde cada linha contém um chunk com:
   - "id": identificador único do chunk
   - "source": caminho do arquivo de origem
   - "chunk_id": número sequencial do chunk dentro do arquivo
   - "text": conteúdo textual do chunk
2) Calcula embeddings dos textos usando SentenceTransformers
   (modelo default: 'sentence-transformers/all-MiniLM-L6-v2').
3) Cria um índice FAISS (Inner Product) e o persiste em `data/index/faiss.index`.
4) Salva também:
   - `data/index/metas.json`: metadados com id/source/chunk_id na mesma ordem dos vetores.
   - `data/index/texts.json`: lista de textos na mesma ordem dos vetores.

Uso
---
    python -m src.embed_index

Pré-requisitos
--------------
- `data/processed/corpus.jsonl` existente (gerado pelo pipeline de ingestão).
- Pacotes: sentence-transformers, faiss, numpy, tqdm.

Saída
-----
- Índice FAISS e arquivos auxiliares salvos em `data/index/`.
- Mensagem final com a quantidade de chunks indexados.
"""

import os, json
import numpy as np
import faiss

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


CORPUS_PATH = "data/processed/corpus.jsonl"
INDEX_DIR = "data/index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_corpus(path):
    """
    Carrega textos e metadados do corpus JSONL.

    Parâmetros
    ----------
    path : str
        Caminho para o arquivo JSONL (um JSON por linha) contendo os chunks.

    Retorna
    -------
    (list[str], list[dict])
        - texts: lista de strings com o conteúdo de cada chunk (na ordem lida).
        - metas: lista de dicionários com metadados mínimos por chunk:
                 {"id": str, "source": str, "chunk_id": int}

    Observações
    -----------
    - A ordem de `texts` e `metas` é preservada para manter alinhamento com os
      vetores de embedding que serão calculados posteriormente.
    - Este método não faz validação profunda do esquema; assume que cada linha
      possui ao menos as chaves "id", "source", "chunk_id" e "text".
    """
    
    texts, metas = [], []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            metas.append(
                {"id": d["id"], 
                 "source": d["source"], 
                 "chunk_id": d["chunk_id"]
                }
            )

    return texts, metas

if __name__ == "__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)
    texts, metas = load_corpus(CORPUS_PATH)

    model = SentenceTransformer(EMB_MODEL)
    batch = 256
    embs = []

    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        x = model.encode(texts[i:i+batch], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        embs.append(x)
    
    embs = np.vstack(embs).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "metas.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    with open(os.path.join(INDEX_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)

    print(f"Index pronto: {len(texts)} chunks")


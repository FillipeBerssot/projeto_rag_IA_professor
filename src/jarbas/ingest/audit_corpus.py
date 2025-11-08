"""
audit_corpus.py
================
Script de auditoria do corpus processado em JSONL.

O que faz
---------
- Lê o arquivo `data/processed/corpus.jsonl` (um JSON por linha, gerado pela ingestão).
- Para cada linha (chunk), extrai:
  - `source`: caminho relativo do arquivo de origem.
  - A extensão do arquivo (`.md`, `.pdf`, `.html`, etc.).
  - A "raiz" (primeiro diretório abaixo de `data/sources/`), usada para agrupar.
- Acumula contagens por:
  1) **Raiz** (ex.: `dados fastapi`, `dados docker`, …).
  2) **Extensão** (ex.: `.html`, `.md`, `.pdf`, …).
- Exibe um relatório no stdout com:
  - Ranking de **chunks por raiz** com um exemplo de caminho por raiz.
  - Ranking de **chunks por extensão**.

Pré-requisitos
--------------
- O arquivo `data/processed/corpus.jsonl` deve existir e conter um JSON por linha com, no mínimo,
  a chave `"source"` (caminho salvo durante a ingestão).

Uso
---
    python -m src.audit_corpus

Saída esperada
--------------
- Um sumário de contagens por raiz.
- Um sumário de contagens por extensão.

Notas
-----
- Caso `source` não esteja sob `data/sources/`, a raiz é tratada como `"data"`.
- Este script não altera arquivos; é apenas diagnóstico/inspeção rápida do corpus.
"""

import os, json, collections


CORPUS = "data/processed/corpus.jsonl"

by_root = collections.Counter()
by_ext = collections.Counter()
examples = {}

with open(CORPUS, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        src = d['source']
        ext = os.path.splitext(src)[1].lower()

        if "data/sources" in src:
            root = src.split("data/sources/")[-1].split("/")[0]
        else:
            root = "data"

        by_root[root] += 1
        by_ext[ext] += 1
        examples.setdefault(root, src)

print("== Chunks por raiz ==")
for k, v in by_root.most_common():
    print(f"{k:30s} {v:8d} e.g {examples[k]}")

print("\n== Chunks por extensão ==")
for k, v in by_ext.most_common():
    print(f"{k or '(sem ext)':10s} {v:8d}")
---
title: Projeto Rag IA Professor
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Pipeline RAG para aprender na prática
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

# Jarbas RAG (Projeto de Estudo Pessoal)

> **Aviso**  
> Este repositório é um **projeto de estudo** feito para praticar RAG (Retrieval-Augmented Generation), Python e Streamlit.  
> **Não** é um produto, não tem garantias de estabilidade e **pode falhar** (por exemplo, com mensagens como `Killed` em máquinas com pouca memória).  
> O objetivo é **aprender**: desmontar, testar, quebrar, consertar e entender cada peça do pipeline.
> Você pode gerar respostas **com OpenAI** (janela grande e mais qualidade) **ou 100% local** (modelo leve Qwen 1.5B), sempre **nos seus próprios dados** indexados com FAISS.

---

## Sumário
- [Visão Geral](#visão-geral)
- [Como o Jarbas funciona (para leigos)](#como-o-jarbas-funciona-para-leigos)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Modelos de Geração: OpenAI vs Local](#modelos-de-geração-openai-vs-local)
- [🚀 Guia rápido](#-guia-rápido)
  - [0) Preparar ambiente](#0-preparar-ambiente)
  - [1) Colocar seus dados](#1-coloque-seus-dados)
  - [2) Ingestão (normalizar e quebrar em chunks)](#2-ingestão-normalizar-e-quebrar-em-chunks)
  - [3) Indexação (embeddings + FAISS)](#3-indexação-embeddings--faiss)
  - [4) Subir a interface (Streamlit)](#4-subir-a-interface)
- [🧠 O que acontece por baixo do capô](#-o-que-acontece-por-baixo-do-capô)
  - [Recuperação](#recuperação-valendo-para-ambos-backends)
  - [Geração (Local vs OpenAI)](#geração-duas-formas)
  - [Por que essas bibliotecas?](#por-que-escolhemos-essas-bibliotecas)
- [⚙️ Parâmetros e decisões de segurança](#️-parâmetros-e-decisões-de-segurança)
- [💡 Dicas de uso](#-dicas-de-uso)
- [Uso no Streamlit](#uso-no-streamlit)
- [🧯 Troubleshooting](#-troubleshooting)
  - [“Killed” / OOM](#killed-servidor-é-encerrado)
  - [CUDA OOM](#runtimeerror-cuda-out-of-memory)
  - [Prompt muito longo](#token-indices-sequence-length--prompt-muito-longo)
  - [Sem OpenAI Key](#faltou-a-openai-key-do-usuário)
  - [Índice ausente](#indexerrorfaiss-no-such-file-or-directory)
- [🧪 Como reproduzir rápido (comandos)](#-como-reproduzir-rápido-comandos)
- [Privacidade & Custos](#privacidade--custos)
- [FAQ](#perguntas-frequentes-faq)
- [📄 Licença & créditos](#-licença--créditos)

---

## Visão Geral

O Jarbas implementa um fluxo **RAG** clássico:

1. **Recuperação** — Localizamos trechos relevantes dos seus documentos com **embeddings** (SentenceTransformers) e **FAISS**.
2. **Montagem de contexto** — Montamos um **prompt** contendo sua **pergunta** e um bloco **Contexto** com os trechos mais similares.
3. **Geração** — Um **modelo de linguagem** (OpenAI *ou* um modelo **local** leve) usa esse contexto para produzir a resposta.
4. **Referências** — Ao final, mostramos **quais trechos** do seu índice foram usados.

O objetivo é ser **didático** (explicações passo a passo, exemplos mínimos) e **prático** (sem depender sempre de APIs pagas).

> ⚠️ **Educação/Estudo:** Este projeto não é um produto profissional.  
> Pode apresentar limitações, especialmente no modo **Local** (modelo pequeno).

---

## Como o Jarbas funciona (para leigos)

Imagine que você tem uma “biblioteca” com os seus PDFs, anotações, arquivos técnicos.  
O Jarbas transforma tudo isso em **números** (chamados *embeddings*) e guarda em um “catálogo” rápido (o **FAISS**).

Quando você faz uma **pergunta**:
- O Jarbas procura no catálogo os **trechos mais parecidos** com a sua pergunta.
- Junta esses trechos e cria um **Contexto**.
- Passa **Pergunta + Contexto** para um **modelo de IA** que escreve uma resposta.
- No final, mostra **de onde** (quais trechos) aquela resposta veio.

Se você escolher **OpenAI**, a IA é mais esperta e tem memória maior.  
Se você escolher **Local**, tudo roda no seu computador, sem internet, mas a IA é **mais simples e limitada**.

---

## 🗂️ Estrutura do projeto

```
.
├─ .streamlit/                 # Config do Streamlit (tema, etc.)
├─ .venv/                      # (opcional) ambiente virtual local
├─ data/
│  ├─ sources/                 # ➜ coloque seus arquivos-fonte aqui (texto)
│  ├─ processed/               # saídas da ingestão (normalizados, chunkados)
│  └─ index/                   # índice FAISS + textos/metadata usados na busca
├─ notebooks/                  # (opcional) experimentos
├─ src/
│  └─ jarbas/
│     ├─ ingest/
│     │  ├─ ingest.py          # 1) normaliza e fatia corpus (→ processed/)
│     │  ├─ embed_index.py     # 2) cria embeddings e índice FAISS (→ index/)
│     │  └─ audit_corpus.py    # (opcional) inspeciona corpus/chunks
│     ├─ rag/
│     │  ├─ local_backend.py   # backend RAG LOCAL (Qwen 2.5 1.5B)
│     │  └─ openai_backend.py  # backend RAG com OpenAI (gpt-4o-mini)
│     └─ utils/
│        └─ text.py            # utilitários p/ limpeza/particionamento de texto
├─ streamlit_app.py            # UI em Streamlit
├─ requirements.txt
├─ README.md
└─ .env                        # (opcional) OpenAI API Key
```

> **Nota sobre os formatos**: o pipeline foi pensado para **texto**. Se você tiver PDFs/HTML/etc., converta para `.txt`/`.md` ou adapte `ingest.py` para o seu caso.

---

## Modelos de Geração: OpenAI vs Local

| Aspecto             | OpenAI (gpt-4o-mini)                      | Local (Qwen 1.5B)                              |
|---------------------|-------------------------------------------|-----------------------------------------------|
| Qualidade           | Alta/estável                               | Básica (modelo pequeno)                        |
| Janela de contexto  | Grande                                     | Menor (≈ 2k tokens)                            |
| Custo               | $$ (precisa de API key)                    | Grátis (roda no seu hardware)                   |
| Privacidade         | Envia prompt à OpenAI                      | 100% local                                      |
| Velocidade          | Varia (rede/latência)                      | Varia (sua CPU/GPU)                             |
| Configurações UI    | Ajustáveis                                 | Travadas para evitar “Killed”/OOM               |

> **Recomendação:** Para respostas longas e robustas, use **OpenAI**.  
> Para testar sem custos e offline, use **Local**.

---

## 🚀 Guia rápido

### 0) Preparar ambiente

```bash
# Python 3.10+ recomendado
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Se for usar a OpenAI, crie um `.env` com sua chave:
```
OPENAI_API_KEY=sk-...
```

### 1) Coloque seus dados
Adicione arquivos de **texto** em `data/sources/`. Exemplos: `.txt`, `.md`.  
(Para outros formatos, converta antes ou ajuste `ingest.py`.)

### 2) Ingestão (normalizar e quebrar em chunks)
```bash
python -m src.jarbas.ingest.ingest
```
**O que acontece:** o script lê `data/sources/`, limpa/normaliza, fatia em **chunks** e grava em `data/processed/` (mais fácil de embutir).

### 3) Indexação (embeddings + FAISS)
```bash
python -m src.jarbas.ingest.embed_index
```
**O que acontece:** calcula **embeddings** (SentenceTransformers) para os chunks de `processed/` e cria o índice **FAISS** em `data/index/` juntamente com `texts.json` e `metas.json`.

> (Opcional) Explore o corpus/chunks com:
> ```bash
> python -m src.jarbas.ingest.audit_corpus
> ```

### 4) Subir a interface
```bash
streamlit run streamlit_app.py
```
Na UI: escolha **Local** (sem chave, mais limitado) ou **OpenAI** (requer API key).  
Escreva sua pergunta e envie.

---

## 🧠 O que acontece por baixo do capô

### Recuperação (valendo para ambos backends)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (leve e rápido) → vetoriza a pergunta e os chunks.
- **Busca**: **FAISS** (index em `data/index/faiss.index`) retorna os `top_k` mais similares.
- **MMR (OpenAI)**: o backend OpenAI ainda aplica **MMR** para trocar parte dos top-k por trechos **diversos**, reduzindo redundância.
- **Contexto**: os trechos selecionados viram um bloco **CONTEXTO** anexado ao *prompt* (com tags `[source :: chunk X]`).

### Geração (duas formas)
- **Local (`src/jarbas/rag/local_backend.py`)**  
  Usa `Qwen/Qwen2.5-1.5B-Instruct` (via `transformers`/`pipeline` com `text-generation`).  
  Como é um **modelo pequeno**, há **limites rígidos** de tamanho do prompt/saída. O código **trunca** pergunta/contexto quando necessário e cita as **Referências** ao final.

- **OpenAI (`src/jarbas/rag/openai_backend.py`)**  
  Envia o *prompt* para `gpt-4o-mini` (ou outro configurado). Aqui a janela é bem maior, mas ainda limitamos o **top_k** e o **tamanho do contexto** (por segurança e custo). Também retorna as **Referências** ao final.

### Por que escolhemos essas bibliotecas?
- **SentenceTransformers**: embeddings de qualidade com custo baixo → perfeito para protótipos.
- **FAISS**: busca vetorial extremamente rápida e madura.
- **Transformers (Hugging Face)**: roda modelos *open* localmente.
- **OpenAI**: alternativa de alta qualidade/estabilidade quando se tem chave.
- **Streamlit**: cria **UI rápida** para testar o RAG sem construir frontend.
- **NumPy / Torch**: base numérica e execução acelerada (CPU/GPU).

---

## ⚙️ Parâmetros e decisões de segurança

- **Local (Qwen 1.5B)**: parâmetros **travados** na UI para evitar `Killed`/OOM.
  - `top_k = 4`, `temperature = 0.2`, `max_output_tokens ~320` (e o backend ainda ajusta dinamicamente).
- **OpenAI**: `top_k` configurável com **limite superior** (clamp) para evitar prompts gigantes.  
  Mesmo com janelas grandes, **muito contexto** pode degradar a qualidade e aumentar custo/latência.

> A UI libera **recursos** ao alternar de Local ↔ OpenAI: fecha pipelines, limpa cache do Streamlit e, se houver GPU, chama `torch.cuda.empty_cache()`.

---

## 💡 Dicas de uso

- **Faça perguntas objetivas** (1–3 frases) e focadas em um tópico.
- Se a resposta local vier fraca, **use o motor OpenAI** (quando possível).
- **Curadoria do corpus** importa: remova lixo, duplicatas e textos não informativos.
- **Chunks menores** (com sobreposição) tendem a recuperar passagens mais precisas.
- Ajuste `ingest.py`/`utils/text.py` para o **seu domínio** (regras de limpeza, splits, metadados).

## Uso no Streamlit

1. **Escolha do motor**
   - **OpenAI (gpt-4o-mini)** — exige **API key** (`sk-...`), pode ajustar `top_k`, `temperatura`, `tokens de saída`.
   - **Local (Qwen 1.5B)** — não exige chave; **parâmetros travados** por segurança.

2. **Escreva a pergunta** e clique **Perguntar**.

3. **Resposta** virá com:
   - explicação didática (resumo, passos, exemplo, dicas);
   - **Referências** listando os *chunks* usados.

> Dica: Em **Local**, faça perguntas **curtas e objetivas**. O modelo tem janela menor e pode truncar entradas muito longas.

---

## 🧯 Troubleshooting

### “Killed” (servidor é encerrado)
- Sintoma típico de **falta de RAM** (ou OOM no container).
- Use o **modo Local** com os **parâmetros travados** que vêm no app.
- No **OpenAI**, evite `top_k` alto; a UI aplica **clamp** automático.
- Feche outras abas/processos pesados; em GPU use `nvidia-smi` para checar uso.

### `RuntimeError: CUDA out of memory`
- Reduza o tamanho das perguntas e **top_k**; reinicie o app após alternar motores.
- Em máquinas sem GPU, rode tudo em CPU (o projeto já faz isso automaticamente).

### “Token indices sequence length …” / prompt muito longo
- O backend **trunca** pergunta/contexto, mas se insistir: reduza `top_k` e seja mais direto.

### “Faltou a OpenAI Key do Usuário.”
- Preencha a chave em **Configurações** (ou `.env`), iniciando com `sk-`.

### “IndexError/FAISS: no such file or directory”
- Rode **na ordem correta**: `ingest.py` ➜ `embed_index.py` ➜ `streamlit_app.py`.

---

## 🧪 Como reproduzir rápido (comandos)

```bash
# 0) ambiente
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) dados (adicione seus .txt/.md em data/sources/)

# 2) ingestão
python -m src.jarbas.ingest.ingest

# 3) índice vetorial
python -m src.jarbas.ingest.embed_index

# (opcional) auditoria
python -m src.jarbas.ingest.audit_corpus

# 4) UI
streamlit run streamlit_app.py
```

---

## Privacidade & Custos

- **Local**: tudo roda no seu computador. Sem envios externos.
- **OpenAI**: o prompt (pergunta + contexto) é enviado à OpenAI. Você paga **por token** de entrada/saída.
- **Chave**: a API key é solicitada **apenas** quando você escolhe OpenAI no Streamlit e é usada **somente na sessão**.

---

## Perguntas Frequentes (FAQ)

**1) Por que minha resposta foi curta/incompleta no Local?**  
O Qwen 1.5B tem **janela menor**. O backend aplica **truncamentos**. Tente encurtar a pergunta ou use o motor **OpenAI**.

**2) O que é `top_k`?**  
É o número de **trechos** do seu índice enviados no **Contexto**. Mais trechos = mais fatos, mas também **mais tokens** e custo/latência (OpenAI).

**3) O que causa o “Killed”?**  
Geralmente **falta de memória** (RAM/GPU) quando a entrada fica grande demais. O app já limita isso, mas use valores conservadores.

**4) Posso usar outro modelo local?**  
Sim. Ajuste `GEN_MODEL` no `local_backend.py` e assegure-se de que a **janela** e **VRAM** comportam o modelo.

**5) Posso usar outros provedores além da OpenAI?**  
A arquitetura permite, mas você precisará implementar um backend análogo (`*_backend.py`) para o provedor desejado.

---
## 📄 Licença & créditos

Projeto feito para **fins educacionais**. 
Código liberado sob licença **MIT** 
Modelos e pacotes externos seguem as **suas próprias licenças** (consulte os repositórios).

Feito por **Fillipe Berssot** como **projeto de estudo**.  
Ideias e ajustes de prompt/pipeline foram inspirados pela documentação das libs usadas e por boas práticas comuns em RAG.
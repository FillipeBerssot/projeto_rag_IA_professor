"""
openai_backend.py
=============
Pipeline RAG usando **embeddings locais** (SentenceTransformers + FAISS) para
recuperação e a **API da OpenAI** para geração de textos.

Fluxo resumido
--------------
1. Carrega o índice FAISS e os metadados/textos correspondentes de `data/index/`.
2. Gera o embedding da consulta do usuário e faz busca vetorial no FAISS.
3. Reclassifica candidatos com MMR para promover diversidade.
4. Monta o *contexto* com trechos (chunks) recuperados e preenche um *prompt* didático.
5. Envia o prompt para o modelo da OpenAI (ex.: `gpt-4o-mini`) e retorna a resposta.
6. Anexa, ao final, as referências dos trechos usados no contexto.

Observações
-----------
- Este arquivo é a variante **OpenAI** (geração externa). A variante local (FLAN-T5)
  fica em `rag.py`.
- A segurança da *API key* é responsabilidade do chamador (ex.: leitura via Streamlit).
"""

import os, json
import faiss
import numpy as np

from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# Prompt para perguntas complexas:
PROMPT_TEMPLATE = """\
Você é **Jarbas**, um professor especialista e didático de programação.
Seu foco principal: **Python (do zero ao avançado), FastAPI, Docker, Git/GitHub, Análise/Ciência de Dados (Pandas, NumPy, Matplotlib), Pydantic, SQL, PostgreSQL**.
Idioma: **português do Brasil (pt_BR)**.

### Objetivo
Ensinar com clareza e progressão pedagógica. Responda **passo a passo**, com exemplos mínimos e testáveis, explicando o **porquê** de cada etapa. Adapte a explicação para alguém que está estudando, mas sem superficialidade técnica.

### Regras de Uso do Contexto (RAG)
- Use **apenas** o conteúdo em **Contexto** como referência factual.  
- Se a resposta exigir algo **não presente no contexto**, **diga explicitamente** que não há informação suficiente e **sugira próximos passos** (ex.: “considere consultar X / buscar Y”).
- **Nunca** invente APIs, parâmetros, sintaxes ou resultados que não estejam no contexto ou que você não tenha certeza.
- Se houver **conflitos** no contexto, **explique o conflito** e sugira a fonte mais confiável.

### Estilo Didático (sempre)
- Tom: cordial, motivador, direto; **sem jargão desnecessário**.
- Estruture em **seções** com títulos curtos.
- Use **listas numeradas** para passos e **bullets** para dicas/observações.
- Inclua **exemplos mínimos executáveis** (quando fizer sentido).
- Ao final, inclua **“Teste rápido”** (1–3 itens) e **“Erros comuns”** (até 3).
- Se for um processo, inclua **“Checklist de verificação”** curto.

### Código e Formatação
- Em trechos de código, use blocos com linguagem (```python, ```bash, ```sql).
- Prefira exemplos **curtos e completos**.  
- Comente linhas importantes nos exemplos (`# por quê`).  
- Em SQL, destaque chaves/joins e explique resultado esperado.  
- Em Docker, diferencie **build** vs **run** e explique portas/volumes/envs.  
- Em Pandas, mostre `.head()` esperado quando pertinente.  
- Em FastAPI, mostre app mínimo + rota + validação Pydantic.  
- Em Git, mostre comandos e **quando** usá-los.

### Verificação de Entendimento
- Inclua **1 pergunta de checagem** (ex.: “Você consegue explicar por que usamos X antes de Y?”).  
- Se o tema for longo, proponha **rota de estudo** com 2–4 etapas progressivas.

### Segurança e Responsabilidade
- **Não** revele raciocínio interno passo a passo (“cadeia de pensamento”). Mostre apenas o **resultado** e as **justificativas concisas**.
- Se for tema sensível ou fora do escopo técnico, **recuse** com gentileza e redirecione.

### Estrutura esperada da resposta
1) **Resumo bem estruturado**
2) **Passo a passo bem estruturado e completo**
3) **Exemplo(s) mínimo(s)**
4) **Dicas & Observações**
5) **Teste rápido**
6) **Erros comuns**
7) **Próximos passos** (se faltar contexto)
8) **Referências (do contexto)** – mantenha a lista fornecida ao final do output principal.

### Pergunta
{question}

### Contexto
{context}

### Responda agora conforme as instruções acima.
"""

# Prompt para perguntas curtas:
PROMPT_TEMPLATE_COMPACT = """\
Você é Jarbas, professor didático (pt_BR). Responda passo a passo, com exemplo mínimo executável e 1 teste rápido.
Use somente o **Contexto**; se faltar, admita e sugira próximos passos.

Pergunta: {question}

Contexto:
{context}

Responda agora:
"""


INDEX_DIR = "data/index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"

MAX_INPUT_TOKENS = 100_000

# Orçamentos de segurança para prompt no OpenAI
# (aprox. tokens por ~chars; 4 chars ≈ 1 token)
INPUT_TOKEN_BUDGET = 12000          # prompt completo (pergunta + contexto + instruções)
OUTPUT_TOKEN_BUDGET = 1200          # limite de saída
MAX_CONTEXT_CHARS   = 40_000         # guarda-chuva extra por caracteres
MAX_CHARS_PER_BLOCK = 2_000          # cada chunk é truncado antes de entrar no prompt

def approx_tokens(s: str) -> int:
    """
    Estima a contagem de tokens dado um texto em caracteres.

    Regra aproximada
    ----------------
    ~4 caracteres ≈ 1 token.

    Parâmetros
    ----------
    s : str
        Texto de entrada.

    Retorna
    -------
    int
        Número aproximado de tokens (>= 1).
    """
        
    return max(1, len(s) // 4)

def truncate_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars]

class RAGTeacherOpenAI:
    """
    Orquestrador RAG (recuperação + geração) usando embeddings locais
    e a API de geração de texto da OpenAI.

    Responsabilidades
    -----------------
    - Carregar índice FAISS e metadados/textos.
    - Codificar consultas e recuperar vizinhos (similaridade).
    - Reordenar candidatos com MMR (opcional) para reduzir redundância.
    - Preparar prompt didático e enviar para OpenAI.
    - Devolver resposta com referências dos trechos utilizados.

    Parâmetros do construtor
    ------------------------
    index_dir : str
        Diretório contendo `faiss.index`, `metas.json` e `texts.json`.
    top_k : int
        Quantidade de trechos a retornar (após MMR) para compor o contexto.
    api_key : Optional[str]
        Chave de API do usuário para autenticar no endpoint da OpenAI.
    """

    def __init__(self, index_dir: str = INDEX_DIR, top_k: int = 5, api_key: Optional[str] = None):
        self.top_k = top_k

        # Validação da chave
        if not api_key:
            raise RuntimeError("Faltou a OpenAI Key do Usuário.")
        
        # Carrega FAISS + dados
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "texts.json"), "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(os.path.join(index_dir, "metas.json"), "r", encoding="utf-8") as f:
            self.metas = json.load(f)
        
        # Embeddings
        self.emb = SentenceTransformer(EMB_MODEL, device="cpu")

        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL

    @staticmethod
    def _cuda_available() -> bool:
        """
        Verifica se CUDA está disponível (uso opcional em chamadas externas).

        Retorna
        -------
        bool
            `True` se `torch.cuda.is_available()` for verdadeiro; `False` caso contrário.
        """

        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def retrieve(self, query: str) -> List[Tuple[float, int]]:
        """
        Recupera `top_k` vizinhos mais similares no FAISS para a consulta fornecida.

        Etapas
        ------
        1. Gera o embedding normalizado da `query` com SentenceTransformers.
        2. Faz busca de similaridade (inner product) no índice FAISS.
        3. Retorna a lista `(score, idx)`.

        Parâmetros
        ----------
        query : str
            Texto da consulta do usuário.

        Retorna
        -------
        list[tuple[float, int]]
            Lista de tuplas `(score, index_no_faiss)`.
        """

        q = self.emb.encode([query], convert_to_numpy=True, 
                            normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, self.top_k)

        return list(zip(scores[0].tolist(), idxs[0].tolist()))
    
    def build_context_blocks(self, hits: List[Tuple[float, int]]) -> List[str]:
        """
        Constrói blocos de contexto formatados com *tag* de origem e o texto do chunk.

        Formato
        -------
        Cada bloco retornado é uma string como:
        ```
        [<source> :: chunk <id>]
        <conteúdo do trecho>
        ```

        Parâmetros
        ----------
        hits : list[tuple[float, int]]
            Pares `(score, idx)` retornados pela recuperação/seleção.

        Retorna
        -------
        list[str]
            Blocos de contexto prontos para concatenar no prompt.
        """

        blocks = []

        for _, i in hits:
            meta = self.metas[i]
            tag = f"[{meta['source']} :: chunk {meta['chunk_id']}]"
            body = truncate_chars(self.texts[i], MAX_CHARS_PER_BLOCK)
            blocks.append(f"{tag}\n{body}")
        return blocks
    
    def _mmr(self, doc_embs, query_emb, k=7, lambda_=0.5):
        """
        Maximum Marginal Relevance (MMR) para selecionar subconjunto de documentos
        equilibrando **relevância** (similiaridade com a query) e **diversidade**
        (reduz redundância entre documentos selecionados).

        Parâmetros
        ----------
        doc_embs : np.ndarray
            Matriz (N x D) de embeddings dos documentos candidatos.
        query_emb : np.ndarray
            Vetor (D,) de embedding da consulta.
        k : int
            Quantidade de itens a selecionar.
        lambda_ : float
            Peso de relevância (↑) vs diversidade (↓). 1.0 = só relevância.

        Retorna
        -------
        list[int]
            Índices dos `k` documentos escolhidos (na ordem de seleção).
        """

        def norm(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
        D = norm(doc_embs)
        q = norm(query_emb.reshape(1, -1))[0]
        rel = D @ q
        S = D @ D.T

        selected, candidates = [], list(range(len(D)))
        while len(selected) < min(k, len(candidates)):
            if not selected:
                best = int(np.argmax(rel[candidates]))
                selected.append(candidates.pop(best))
                continue

            red = np.max(S[np.ix_(candidates, selected)], axis=1)
            score = lambda_ * rel[candidates] - (1 - lambda_) * red
            best = int(np.argmax(score))
            selected.append(candidates.pop(best))

        return selected
    
    def retrieve_mmr(self, query: str, topn: int = 60, topk: int = 7, lambda_: float = 0.5):
        """
        Recupera `topn` candidatos por FAISS e aplica MMR para escolher `topk`.

        Parâmetros
        ----------
        query : str
            Consulta do usuário.
        topn : int
            Quantidade inicial de candidatos puxados do FAISS.
        topk : int
            Quantidade final de trechos retornados após MMR.
        lambda_ : float
            Peso relevância/diversidade no MMR.

        Retorna
        -------
        list[tuple[float, int]]
            Lista `(score, idx_original_no_faiss)` dos itens escolhidos pelo MMR.
        """

        topn = min(topn, len(self.texts))
        q = self.emb.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, topn)
        cand_idxs = idxs[0].tolist()
        cand_texts = [self.texts[i] for i in cand_idxs]
        cand_embs = self.emb.encode(cand_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        pick = self._mmr(cand_embs, q[0], k=topk, lambda_=lambda_)
        return [(scores[0][p], cand_idxs[p]) for p in pick]
    
    def ask(self, question: str, temperature: float = 0.2, max_output_tokens: int = 600) -> str:
        """
        Executa o ciclo completo: recupera, monta contexto, gera resposta e anexa referências.

        Parâmetros
        ----------
        question : str
            Pergunta do usuário.
        temperature : float
            Temperatura de geração (0.0 = mais determinístico).
        max_output_tokens : int
            Limite aproximado de tokens de saída retornados pelo modelo.

        Retorna
        -------
        str
            Resposta gerada + bloco de referências (fontes do contexto).
        """
        
        q = (question or "").strip()
        if len(q) < 5:
            return "Oi! Diga um tópico (ex: Python básico, FastAPI, Docker, Git/GitHub, Pandas, scikit-learn) para eu ajudar melhor."
        
        hits = self.retrieve_mmr(q, topn=60, topk=self.top_k, lambda_=0.5)
        blocks = self.build_context_blocks(hits)
        
        tpl = PROMPT_TEMPLATE if len(q) > 30 else PROMPT_TEMPLATE_COMPACT

        kept, ctx_chars = [], 0
        for blk in blocks:
            candidate_ctx = "\n\n---\n\n".join(kept + [blk])

            if len(candidate_ctx) > MAX_CONTEXT_CHARS:
                break

            test_prompt = tpl.format(question=q, context=candidate_ctx)
            if approx_tokens(test_prompt) <= INPUT_TOKEN_BUDGET:
                kept.append(blk)
                ctx_chars += len(blk)
            else:
                break

        if not kept and blocks:
            blk0 = truncate_chars(blocks[0], min(MAX_CHARS_PER_BLOCK, MAX_CONTEXT_CHARS))
            candidate_ctx = blk0
            test_prompt = tpl.format(question=q, context=candidate_ctx)

            while approx_tokens(test_prompt) > INPUT_TOKEN_BUDGET and len(blk0) > 200:
                blk0 = blk0[: int(len(blk0) * 0.8)]
                candidate_ctx = blk0
                test_prompt = tpl.format(question=q, context=candidate_ctx)
            kept = [blk0]

        ctx_joined = "\n\n---\n\n".join(kept)
        prompt = tpl.format(question=q, context=ctx_joined)
        prompt += "Resposta: "


        # Chamada a Reponses API
        resp = self.client.responses.create(
            model = self.model,
            input = prompt,
            temperature = temperature,
            max_output_tokens = min(max_output_tokens, OUTPUT_TOKEN_BUDGET),
        )

        out = resp.output_text

        # Montar referencias só com os blocos que couberam
        used_refs = []
        used_context = ctx_joined
        for score, i in hits:
            meta = self.metas[i]
            tag = f"[{meta['source']} :: chunk {meta['chunk_id']}]"

            if tag in used_context:
                used_refs.append(f"- {meta['source']} (chunk {meta['chunk_id']})")

        refs_text = "\n".join(used_refs) if used_refs else "- (nenhuma)"
        return f"{out}\n\n---\nReferências:\n{refs_text}"
    
    def release(self):
        try:
            del self.emb
        except Exception:
            pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
if __name__ == "__main__":
    try:
        user_key = input("Informe sua OpenAI API Key (sk-...): ").strip()
        if not user_key.startswith("sk-"):
            raise RuntimeError("API Key inválida. Deve começar com 'sk-'.")
        
        rag = RAGTeacherOpenAI(top_k=7, api_key=user_key)

        while True:
            q = input("\nOlá, eu sou o Jarbas seu professor de python, faça uma pergunta (ou digite 'sair'): ").strip()
            if q.lower() == "sair":
                break
                
            ans = rag.ask(q, temperature=0.2, max_output_tokens=700)
            print("\n=== Resposta ===\n", ans)
    except KeyboardInterrupt:
        pass
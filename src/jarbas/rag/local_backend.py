"""
local_backend.py
================
Pipeline RAG **100% local** usando:
- Recuperação: SentenceTransformers + FAISS
- Geração: Qwen2.5-1.5B-Instruct (transformers, text-generation)

Notas:
- Qwen é um modelo *causal* (não seq2seq). Usamos `apply_chat_template` para
  montar o prompt no formato de chat.
- A janela típica desta variante é ~2048 tokens. Este arquivo limita a
  entrada (pergunta + contexto) e a saída de forma segura.
"""

import os, json
import faiss
import torch
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline


# Prompt (curto e robusto para modelos pequenos)
PROMPT_SYSTEM = (
    """
    Você é Jarbas, um **professor de programação Python** (pt-BR) **direto e didático**. Sua prioridade é fornecer a informação correta e o exemplo de código **mais simples possível**.

    ---

    **Instruções de Conteúdo e Formato:**
    * Responda com um **resumo conciso**, seguido pelo **código de exemplo** e **explicações breves**.
    * **Seja extremamente conciso e vá direto ao ponto.**
    * Use SOMENTE o bloco CONTEXTO para fatos. Se algo não estiver no contexto, diga que a informação é insuficiente e pare a resposta.
    * **Não invente** APIs, resultados ou fatos.
    * Mantenha um tom encorajador e profissional.
    """
)

PROMPT_USER_TEMPLATE = (
    "Pergunta:\n{question}\n\n"
    "Contexto:\n{context}\n\n"
    "Resposta:"
)

# Caminhos/modelos
INDEX_DIR = "data/index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # recomendado

# Orçamentos de tokens
# Qwen 1.5B costuma ter máx. 2048
MODEL_MAX_TOKENS = 2048
MAX_INPUT_TOKENS = 1600          # pergunta + contexto + instruções do sistema
MAX_NEW_TOKENS  = 384            # saída padrão
MAX_QUESTION_TOKENS = 256        # limite da pergunta dentro da janela


# Utilidades de tokens 
def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_to_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


class RAGTeacher:
    """
    Orquestrador RAG **local** (sem OpenAI):
    - Recuperação: embeddings com SentenceTransformers + FAISS.
    - Geração: Qwen2.5-1.5B-Instruct via `transformers` (text-generation).
    """

    def __init__(self, index_dir: str = INDEX_DIR, top_k: int = 5):
        self.top_k = top_k

        # FAISS + metadados
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "texts.json"), "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(os.path.join(index_dir, "metas.json"), "r", encoding="utf-8") as f:
            self.metas = json.load(f)

        # Embeddings
        emb_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb = SentenceTransformer(EMB_MODEL, device=emb_device)

        # Tokenizer + Gerador (Qwen)
        # trust_remote_code costuma ser necessário para apply_chat_template de alguns modelos
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
        self.generator = pipeline(
            "text-generation",
            model=GEN_MODEL,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # EOS id (para Qwen)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if self.tokenizer.pad_token_id is None and self.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.eos_token_id

    def release(self):
        """Libera memória do gerador/tokenizer e esvazia cache da GPU."""
        try:
            if hasattr(self, "generator"):
                # pipeline -> pode ter .model, .tokenizer, etc.
                try:
                    del self.generator.model
                except Exception:
                    pass
                try:
                    del self.generator.tokenizer
                except Exception:
                    pass
                del self.generator
        except Exception:
            pass
        try:
            if hasattr(self, "tokenizer"):
                del self.tokenizer
        except Exception:
            pass

        # Opcional: embeddings também consomem RAM (não GPU). Normalmente você pode mantê-los,
        # mas se quiser ser agressivo:
        # try:
        #     del self.emb
        # except Exception:
        #     pass

        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Recuperação 
    def retrieve(self, query: str) -> List[Tuple[float, int]]:
        q = self.emb.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q, self.top_k)
        return list(zip(scores[0].tolist(), idxs[0].tolist()))

    def build_context_blocks(self, hits: List[Tuple[float, int]]) -> List[str]:
        blocks: List[str] = []
        for _, i in hits:
            meta = self.metas[i]
            tag = f"[{meta['source']} :: chunk {meta['chunk_id']}]"
            blocks.append(f"{tag}\n{self.texts[i]}")
        return blocks

    # Montagem do prompt (chat)
    def _apply_chat(self, system_text: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _fit_prompt(self, question: str, blocks: List[str]) -> tuple[str, str, dict]:
        """
        Encaixa (pergunta + contexto) dentro de MAX_INPUT_TOKENS.
        Retorna: (prompt_text, used_context, info)
        """
        info = {"question_truncated": False, "reduced_blocks": 0}

        # 1) trunca pergunta
        q_trunc = truncate_to_tokens(self.tokenizer, question, MAX_QUESTION_TOKENS)
        if q_trunc != question:
            info["question_truncated"] = True

        # 2) adiciona blocos até caber
        kept: List[str] = []
        for blk in blocks:
            test_ctx = "\n\n---\n\n".join(kept + [blk])
            user_text = PROMPT_USER_TEMPLATE.format(question=q_trunc, context=test_ctx)
            test_prompt = self._apply_chat(PROMPT_SYSTEM, user_text)
            if count_tokens(self.tokenizer, test_prompt) <= MAX_INPUT_TOKENS:
                kept.append(blk)
            else:
                break

        # 3) se nada coube, força 1 bloco truncado
        if not kept and blocks:
            head = self._apply_chat(PROMPT_SYSTEM, PROMPT_USER_TEMPLATE.format(question=q_trunc, context=""))
            budget = MAX_INPUT_TOKENS - count_tokens(self.tokenizer, head)
            if budget > 0:
                blk0 = truncate_to_tokens(self.tokenizer, blocks[0], max(16, budget))
                kept = [blk0]

        used_ctx = "\n\n---\n\n".join(kept)
        user_text = PROMPT_USER_TEMPLATE.format(question=q_trunc, context=used_ctx)
        prompt = self._apply_chat(PROMPT_SYSTEM, user_text)

        info["reduced_blocks"] = max(0, len(blocks) - len(kept))
        return prompt, used_ctx, info

    # Geração
    def ask(self, question: str, temperature: float = 0.2, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        q = (question or "").strip()
        if len(q) < 5:
            return ("Oi! Diga um tópico (ex.: Python básico, FastAPI, Docker, Git/GitHub, "
                    "Pandas) para eu ajudar melhor.")

        hits = self.retrieve(q)
        blocks = self.build_context_blocks(hits)

        prompt, used_context, fit_info = self._fit_prompt(q, blocks)

        # Janela total do modelo
        input_tok = count_tokens(self.tokenizer, prompt)
        headroom = max(64, MODEL_MAX_TOKENS - input_tok - 8)
        safe_new = max(16, min(int(max_new_tokens), headroom))

        try:
            result = self.generator(
                prompt,
                max_new_tokens=safe_new,
                do_sample=True,
                temperature=float(temperature),
                eos_token_id=self.eos_token_id,
                return_full_text=False,
            )
            out = result[0]["generated_text"]
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                return (
                    "⚠️ **Limite de hardware/janela do modelo local atingido**.\n\n"
                    "- Reduza o tamanho da pergunta.\n"
                    "- Diminua *top_k* (menos contexto).\n"
                    "- Ou selecione o backend **OpenAI** para perguntas longas.\n"
                )
            raise

        # Referências (apenas as que entraram no prompt)
        used_refs = []
        for _, i in hits:
            meta = self.metas[i]
            tag = f"[{meta['source']} :: chunk {meta['chunk_id']}]"
            if tag in used_context:
                used_refs.append(f"- {meta['source']} (chunk {meta['chunk_id']})")
        refs_text = "\n".join(used_refs) if used_refs else "- (nenhuma)"

        # Notas de truncamento
        notes = []
        if fit_info.get("question_truncated"):
            notes.append("• A pergunta foi **encurtada** para caber na janela do modelo local.")
        if fit_info.get("reduced_blocks", 0) > 0:
            notes.append(f"• O contexto foi **reduzido** (−{fit_info['reduced_blocks']} bloco(s)).")
        notes_text = ("\n\n_" + "\n".join(notes) + "_") if notes else ""

        return f"{out}{notes_text}\n\n---\nReferências:\n{refs_text}"


if __name__ == "__main__":
    rag = RAGTeacher(top_k=5)
    while True:
        try:
            q = input("\nPergunte (ou 'sair'): ").strip()
            if q.lower() == "sair":
                break
            ans = rag.ask(q, temperature=0.2, max_new_tokens=MAX_NEW_TOKENS)
            print("\n=== Resposta ===\n", ans)
        except KeyboardInterrupt:
            break

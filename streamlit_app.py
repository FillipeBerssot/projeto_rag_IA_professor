import time
import streamlit as st

from src.jarbas.rag.openai_backend import RAGTeacherOpenAI
from src.jarbas.rag.local_backend import RAGTeacher


# Constantes de Segurança para o Modo Local
LOCAL_TOP_K = 4
LOCAL_TEMPERATURE = 0.2
LOCAL_MAX_OUTPUT_TOKENS = 320

OPENAI_DEFAULT_TOP_K = 5
SAFE_TOPK_MAX_OPENAI = 7

# Configuração da Página
st.set_page_config(
    page_title="Jarbas • O Seu Professor de Programação - Python",
    page_icon="🤖",
    layout="wide",
)

# estado de sessão para saber quando o motor muda 
if "active_engine" not in st.session_state:
    st.session_state.active_engine = None
if "local_instance" not in st.session_state:
    st.session_state.local_instance = None 

# Sidebar: Configurações
with st.sidebar:
    st.header("⚙️ Configurações")

    engine = st.selectbox(
        "Motor de geração",
        ["OpenAI (gpt-4o-mini)", "Local (Qwen/Qwen2.5-0.5B-Instruct)"],
        help = (
            "• OpenAI: janela maior, respostas mais completas, requer API key.\n"
            "• Local (Qwen2.5 - 0.5B): roda no seu hardware, sem chave; é mais limitado "
            "em contexto e saída."
        ) ,
    )

    # OPENAI: mostrar campos editáveis
    if engine.startswith("OpenAI"):
        user_api_key = st.text_input(
            "Sua OpenAI API KEY",
            type="password",
            placeholder="sk-...",
            help="Sua chave é usada apenas nesta sessão.",
        )

        top_k = st.slider(
            "top_k (trechos recuperados)",
            1, 10, OPENAI_DEFAULT_TOP_K,
            help=(
                f"Quantos trechos do seu índice entram no CONTEXTO. "
                f"Valores maiores ↑ trazem mais fatos, mas aumentam custo/latência e podem encerrar a sessão ('Killed'). "
                f"Recomendado ≤ {SAFE_TOPK_MAX_OPENAI}"
            ),
        )

        # Clamp + aviso
        effective_top_k = min(top_k, SAFE_TOPK_MAX_OPENAI)
        if top_k > SAFE_TOPK_MAX_OPENAI:
            st.warning(f"Para estabilidade, limitei o top_k de {top_k} -> {effective_top_k}")

        temperature = st.slider(
            "temperatura",
            0.0, 1.0, 0.2, 0.1,
            help=(
                "Controle de criatividade da geração. 0.0 = mais determinístico; 1.0 = mais criativo. "
                "Para respostas técnicas, 0.1–0.3 costuma funcionar bem."
            ),
        )

        max_output_tokens = st.slider(
            "tokens de saída (aprox.)",
            64, 4000, 700, 64,
            help=(
                "Limite de tokens gerados na RESPOSTA. Aumentar permite respostas mais longas, "
                "mas custa mais (OpenAI) e pode ficar verboso."
            ),
        )

    # LOCAL: travar opções (valores fixos)
    else:
        user_api_key = None  # não usado
        # Mostrar como desabilitado, só para o usuário ver o que está valendo:
        st.slider(
            "top_k (trechos recuperados)",
            1, 10, LOCAL_TOP_K, disabled=True,
            help=(
                "No modo LOCAL os parâmetros ficam travados para evitar 'Killed' e estouros de memória. "
                f"Valor fixo: {LOCAL_TOP_K}."
            ),
        )
        st.slider(
            "temperatura",
            0.0, 1.0, LOCAL_TEMPERATURE, 0.1, disabled=True,
            help="Travado no modo LOCAL. Valor informativo.",
        )
        st.slider(
            "tokens de saída (aprox.)",
            64, 2000, LOCAL_MAX_OUTPUT_TOKENS, 64, disabled=True,
            help=(
                "Travado no modo LOCAL. O backend também limita dinamicamente para caber na janela do modelo."
            ),
        )

        # Instruções extras do modo local
        st.info(
            "🖥️ **Modo Local (Qwen2.5 - 0.5B)**\n\n"
            "- Ideal para testes sem API key.\n"
            "- Contexto curto (janela menor). Perguntas e respostas muito longas podem ser encurtadas.\n"
            "- Evite colar textos gigantes na pergunta.\n"
            "- Se precisar de respostas mais extensas, selecione o motor **OpenAI**."
        )

# detectar mudança de motor e limpar recursos pesados
def _cleanup_on_switch():
    # se tínhamos um local carregado, liberar recursos
    loc = st.session_state.get("local_instance")
    if loc is not None:
        try:
            loc.release()
        except Exception:
            pass
        st.session_state.local_instance = None
    # limpar recursos cacheados do streamlit (inclusive modelos)
    st.cache_resource.clear()
    # coletor + GPU
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# engine é a variável que você já define na sidebar
if st.session_state.active_engine is None:
    st.session_state.active_engine = engine
elif st.session_state.active_engine != engine:
    _cleanup_on_switch()
    st.session_state.active_engine = engine

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Cache: carregar modelos/índice só uma vez
@st.cache_resource(show_spinner=True)
def load_teacher_local():
    t0 = time.time()
    teacher = RAGTeacher(top_k=LOCAL_TOP_K)
    return teacher, (time.time() - t0)

@st.cache_resource(show_spinner=True)
def load_teacher_openai(top_k_value: int, api_key: str):
    t0 = time.time()
    teacher = RAGTeacherOpenAI(top_k=top_k_value, api_key=api_key)
    return teacher, (time.time() - t0)

# Caixa de entrada
st.subheader("Jarbas • O Seu Professor de Programação - Python")
st.subheader("Faça sua pergunta")

question = st.text_area(
    " ", 
    height=100, 
    placeholder="Escreva aqui...",
    label_visibility="collapsed"
)
answer = None

# Dicas específicas para o modo local (expander na área central)
if not "engine" in locals() or not engine.startswith("OpenAI"):
    with st.expander("💡 Dicas para o modo local (Qwen2.5 - 0.5B)"):
        st.markdown(
            """
- Faça **perguntas objetivas e curtas** (1–3 frases).
- Prefira **tópicos específicos** (ex.: “Como criar uma rota POST no FastAPI com Pydantic?”).
- Evite colar manuais enormes na pergunta — o **contexto** já vem do índice.
- Se a resposta vier incompleta, **refaça** a pergunta de forma mais direta.
- Para conteúdos longos, **use OpenAI** na aba de configurações.
            """
        )       

col_run1, col_run2 = st.columns([1, 2])
with col_run1:
    run = st.button("Perguntar", type="primary")
with col_run2:
    st.write("")

# Execução
if run:
    # Validações antes de carregar/rodar
    if not question or not question.strip():
        st.warning("Escreva uma pergunta antes de continuar.")
        st.stop()

    if not engine.startswith("OpenAI") and len(question) > 1200:
        st.info("Sua pergunta é bem longa. No modo **Local** eu posso encurtá-la para caber na janela do modelo.")

    # Carregamento do backend
    if engine.startswith("OpenAI"):
        if not user_api_key or not user_api_key.strip().startswith("sk-"):
            st.error("Informe sua OpenAI API Key (formato 'sk-...').")
            st.stop()
        with st.spinner("Carregando índice/modelo (OpenAI)..."):
            teacher, load_secs = load_teacher_openai(
                effective_top_k,
                user_api_key.strip()
            )
        st.success(f"Pronto em {load_secs:.2f}s • top_k={effective_top_k}")
    else:
        with st.spinner("Carregando índice/modelo (Local)..."):
            teacher, load_secs = load_teacher_local()
            st.session_state.local_instance = teacher  # manter referência p/ release()
        st.success(f"Pronto em {load_secs:.2f}s • Modo Local • top_k={LOCAL_TOP_K}")

    # Geração
    with st.spinner("Gerando resposta..."):
        try:
            if engine.startswith("OpenAI"):
                answer = teacher.ask(
                    question,
                    temperature=float(temperature),
                    max_output_tokens=int(max_output_tokens),
                )
            else:
                answer = teacher.ask(question)
        except Exception as e:
            st.error(f"Ops! Algo deu errado ao gerar a resposta: {e}")
            answer = None

# Render da resposta (fora do if run, mas protegido)
if answer:
    st.divider()
    st.subheader("Resposta")
    st.write(answer)

    st.divider()
    st.markdown(
        """
        **Sobre este projeto**

        Este é um projeto **educacional** feito para estudo de RAG e interfaces.  
        Não é um produto profissional e **pode apresentar limitações ou erros** — principalmente no modo **Local**, que usa um modelo pequeno e roda no seu hardware.

        Se você precisa de respostas mais completas e estáveis, utilize o motor **OpenAI**.
        """
    )
    st.caption("Jarbas • RAG nos seus dados com geração OpenAI ou local")

    
"""
text_utils.py
==============
Utilidades de pré-processamento de texto usadas no pipeline de ingestão/RAG.

Funções:
- clean_text: normaliza texto bruto removendo caracteres nulos, espaços/quebras em excesso.
- chunk_text: divide um texto longo em blocos (*chunks*) por parágrafos, com sobreposição,
  para melhorar a recuperação via embeddings/FAISS.
"""

import re

from typing import List


def clean_text(s: str) -> str:
    """
    Limpa e normaliza um texto bruto.

    Operações realizadas:
    - Remove caracteres nulos (`\\x00`).
    - `strip()` para remover espaços em branco nas extremidades.
    - Colapsa múltiplos espaços/tabs em um único espaço.
    - Colapsa sequências de 3+ quebras de linha em apenas 2 quebras.

    Parâmetros
    ----------
    s : str
        Texto de entrada (possivelmente contendo ruído, múltiplos espaços/quebras).

    Retorna
    -------
    str
        Texto normalizado, preservando estrutura básica sem excesso de espaços/linhas.
    """

    s = s.replace('\x00', ' ').strip()
    # Colapsa espaços e quebra múltiplas
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    
    return s

def chunk_text(text: str, 
               max_chars: int = 1600, 
               overlap: int = 100
    ) -> List[str]:
    """
    Divide um texto longo em *chunks* por parágrafos, com sobreposição opcional.

    Estratégia:
    - Quebra o texto original em parágrafos (linhas não vazias).
    - Agrega parágrafos sequencialmente até atingir `max_chars`.
    - Ao iniciar um novo bloco, inclui uma "cauda" do bloco anterior
      com `overlap` caracteres para preservar contexto entre chunks.

    Dicas de uso:
    - `max_chars ~ 1200` costuma equilibrar bem contexto e densidade de informação
      para recuperação sem estourar janelas de contexto dos geradores.
    - Ajuste `overlap` para garantir que conceitos/frases não fiquem cortados
      entre dois chunks e a busca mantenha coerência.

    Parâmetros
    ----------
    text : str
        Texto completo a ser dividido.
    max_chars : int, opcional (padrão=1200)
        Tamanho máximo (em caracteres) de cada chunk.
    overlap : int, opcional (padrão=150)
        Quantidade de caracteres finais do chunk anterior a serem repetidos no início
        do próximo (se houver), para manter continuidade.

    Retorna
    -------
    list[str]
        Lista de chunks de texto, na ordem original.
    """

    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks, buf = [], ""
    
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # Inicia novo bloco com sobreposição do final do anterior
            if buf and overlap > 0:
                tail = buf[-overlap:]
                buf = (tail + "\n" + p).strip()
            else:
                buf = p
    
    if buf:
        chunks.append(buf)
        
    return chunks

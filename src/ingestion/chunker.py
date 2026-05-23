"""
src/ingestion/chunker.py
clean text + split into 500-word chunks
"""

import re
from typing import List

def clean_text(text: str) -> str:
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    if not words: return chunks
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - chunk_overlap)
        
    return chunks

def process(text: str) -> List[str]:
    cleaned = clean_text(text)
    return chunk_text(cleaned)

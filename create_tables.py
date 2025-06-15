#!/usr/bin/env python3
"""
Embed code-index CSVs into Neo4j VECTOR indexes (no LanceDB dependency).
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List
import openai

load_dotenv()

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")

MODEL_NAME  = "text-embedding-3-small"
EMB_DIM     = 1536
MAX_TOKENS  = 8000
def embed_batch(texts: List[str]) -> List[List[float]]:
    res = openai.embeddings.create(model=MODEL_NAME, input=texts)
    return [item.embedding for item in res.data]

# TODO: don't clip tokens
def clip_tokens(text: str, limit: int, enc="cl100k_base") -> str:
    enc = tiktoken.get_encoding(enc)
    ids = enc.encode(text)
    return enc.decode(ids[:limit]) if len(ids) > limit else text

def bulk_insert(tx, label: str, rows: List[dict]):
    tx.run(
        f"UNWIND $rows AS r CREATE (n:{label}) SET n = r",
        rows=rows
    )


def ensure_vector_index(tx, label: str):
    tx.run(f"""
        CREATE VECTOR INDEX {label.lower()}_vec
        IF NOT EXISTS
        FOR (n:{label}) ON (n.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {EMB_DIM},
            `vector.similarity_function`: 'cosine'
          }}
        }}
    """)

if len(sys.argv) != 2:
    sys.exit("Usage: python embed_to_neo4j.py <path/to/repo>")

repo_root = Path(sys.argv[1]).resolve()
slug      = repo_root.name
proc_dir  = Path("processed") / slug

method_csv = proc_dir / "method_data.csv"
class_csv  = proc_dir / "class_data.csv"

methods = pd.read_csv(method_csv).fillna("empty")
classes = pd.read_csv(class_csv).fillna("empty")

methods["text"] = methods["source_code"]
classes["text"] = classes.apply(
    lambda r: f"File: {r.file_path}\nClass: {r.class_name}\n\n"
              f"Source Code: {clip_tokens(r.source_code, MAX_TOKENS)}",
    axis=1
)

print("Embedding Method rows …")
methods["embedding"] = embed_batch(methods["text"].tolist())

print("Embedding Class rows …")
classes["embedding"] = embed_batch(classes["text"].tolist())

# md_rows = []
# for p in repo_root.rglob("*"):
#     if p.suffix in {".md", ".sh"} and p.is_file():
#         txt   = clip_tokens(p.read_text(encoding="utf-8", errors="ignore"), MAX_TOKENS)
#         vec   = embed_batch([txt])[0]
#         md_rows.append({
#             "file_path": str(p.relative_to(repo_root)),
#             "text": txt,
#             "embedding": vec,
#             "kind": p.suffix.lstrip(".")
#         })

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

with driver.session() as sess:
    sess.execute_write(ensure_vector_index, "Method")
    sess.execute_write(ensure_vector_index, "Class")
    # if md_rows:
    #     sess.execute_write(ensure_vector_index, "Doc")

    # chunk to avoid huge UNWIND lists
    CHUNK = 500
    for start in range(0, len(methods), CHUNK):
        chunk = methods.iloc[start:start+CHUNK]
        sess.execute_write(bulk_insert, "Method", chunk.to_dict("records"))
    for start in range(0, len(classes), CHUNK):
        chunk = classes.iloc[start:start+CHUNK]
        sess.execute_write(bulk_insert, "Class", chunk.to_dict("records"))
    # for start in range(0, len(md_rows), CHUNK):
    #     sess.execute_write(bulk_insert, "Doc", md_rows[start:start+CHUNK])

driver.close()

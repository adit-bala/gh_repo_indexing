from __future__ import annotations
import os, sys, json, readline, textwrap
from typing import List, Dict
from neo4j import GraphDatabase
import openai

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")

MODEL_NAME = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMB_DIM    = 1536
TOP_K      = int(os.getenv("TOP_K", "15"))
RERANK     = os.getenv("OPENAI_RERANK") == "1"                    # opt-in

def embed(texts: List[str]) -> List[List[float]]:
    """Batch embed via OpenAI embeddings endpoint."""
    res = openai.embeddings.create(model=MODEL_NAME, input=texts)
    return [d.embedding for d in res.data]

def query_neo(q_vec: List[float]) -> List[Dict]:
    """Return TOP_K Method & Class nodes ordered by cosine similarity."""
    cypher = """
    WITH $vec AS q
    MATCH (n)
    WHERE n.embedding IS NOT NULL AND (n:Method OR n:Class)
    WITH n, vector.similarity.cosine(n.embedding, q) AS score
    ORDER BY score DESC LIMIT $k
    RETURN labels(n)[0] AS label, n.file_path AS file, n.class_name  AS clazz,
           n.name     AS method, score
    """
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, vec=q_vec, k=TOP_K)]

def rerank_with_openai(query: str, candidates: List[Dict]) -> List[Dict]:
    """Ask GPT-4 / gpt-3.5 to rerank the candidates by relevance."""
    prompt = "You are a codebase search reranker. Rank the following code " \
             "snippets by their relevance to the user query.\n\n" \
             f"Query:\n{query}\n\nCandidates:\n"
    for i, c in enumerate(candidates, 1):
        snippet = f"{c['label']} {c.get('clazz','')} {c.get('method','')}".strip()
        prompt += f"{i}. File: {c['file']} - {snippet}\n"
    prompt += "\nReturn a JSON array of indices in best-to-worst order."

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content": prompt}],
        temperature=0)
    try:
        order = json.loads(resp.choices[0].message.content)
        return [candidates[i-1] for i in order if 1 <= i <= len(candidates)]
    except Exception:
        # fallback to original order
        return candidates

def smart_print(hit: Dict, rank: int):
    print(f"\n#{rank}  score={hit['score']:.4f}")
    print(f"   file  : {hit['file']}")
    if hit['label'] == "Class":
        print(f"   class : {hit.get('clazz')}")
    else:
        print(f"   class : {hit.get('clazz')}")
        print(f"   method: {hit.get('method')}")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

print("Ready. Type your query (or Ctrl-D to exit):")

try:
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            break
        if not query:
            continue
        q_vec = embed([query])[0]
        hits  = query_neo(q_vec)

        # optional reranking
        if RERANK and hits:
            hits = rerank_with_openai(query, hits)

        for i, h in enumerate(hits, 1):
            smart_print(h, i)

except KeyboardInterrupt:
    pass
finally:
    driver.close()
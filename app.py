#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List

import openai
import requests
from cohere import Client as Cohere
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")
MODEL_NAME = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "15"))
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
COHERE_KEY = os.getenv("COHERE_API_KEY")
co = Cohere(COHERE_KEY) if COHERE_KEY else None
GH_URL = os.getenv("GH_REPO_URL")
GH_PAT = os.getenv("GH_PAT")
DIFF_LINES = int(os.getenv("DIFF_LINES", "200"))

CHAT_SYSTEM_PROMPT = (
    "You are a code search and context gathering specialist. Your goal is to find and provide detailed context about specific code patterns, methods, or classes in the codebase.\n\n"
    "You will be given:\n"
    "1. <code_context> - Relevant code snippets from the repository, including:\n"
    "   - Method and class definitions\n"
    "   - File paths and locations\n"
    "   - Dependencies and references\n"
    "2. <diff> - Recent changes to the codebase\n\n"
    "Your response should:\n"
    "1. Identify all relevant code locations that match the search query\n"
    "2. For each match, provide:\n"
    "   • Full file path and location\n"
    "   • Class and method context\n"
    "   • Key code snippets with line numbers\n"
    "   • Related dependencies or references\n"
    "   • Recent changes if any\n"
    "3. Organize the information by relevance to the query\n"
    "4. Include any additional context that might be helpful\n\n"
    "Format your response as:\n"
    "1. Summary of Found Matches\n"
    "2. Detailed Context for Each Match:\n"
    "   • File: [path]\n"
    "   • Class: [name]\n"
    "   • Method: [name] (if applicable)\n"
    "   • Code:\n"
    "     [relevant code snippet]\n"
    "   • Dependencies:\n"
    "     [related code references]\n"
    "   • Recent Changes:\n"
    "     [relevant diffs if any]\n"
    "3. Additional Context\n"
    "\n"
    "<code_context>{code_context}</code_context>\n\n"
    "<diff>{diff}</diff>"
)

HEADERS_GH = {
    "Authorization": f"Bearer {GH_PAT}" if GH_PAT else None,
    "Accept": "application/vnd.github.v3.diff",
}
HEADERS_GH = {k: v for k, v in HEADERS_GH.items() if v is not None}

def embed(texts: List[str]) -> List[List[float]]:
    resp = openai.embeddings.create(model=MODEL_NAME, input=texts)
    return [d.embedding for d in resp.data]

def query_neo(q_vec: List[float]) -> List[Dict]:
    cypher = """
    WITH $vec AS q
    MATCH (n)
    WHERE n.embedding IS NOT NULL AND (n:Method OR n:Class)
    WITH n, vector.similarity.cosine(n.embedding, q) AS score
    ORDER BY score DESC LIMIT $k
    RETURN labels(n)[0] AS label,
           n.file_path   AS file,
           n.class_name  AS clazz,
           n.name        AS method,
           n.source_code AS source_code,
           score
    """
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, vec=q_vec, k=TOP_K)]

def rerank_with_cohere(query: str, hits: List[Dict]) -> List[Dict]:
    if not co:
        return hits
    documents = [h.get("snippet") or f"{h['clazz']} {h.get('method','')}" for h in hits]
    try:
        rer = co.rerank(model="rerank-english-v3.0", query=query, documents=documents)
        order = [r.index for r in rer]
        return [hits[i] for i in order]
    except Exception:
        return hits

def _owner_repo(url: str):
    path = url.replace("https://github.com/", "").replace(".git", "")
    owner, repo = path.split("/", 1)
    return owner, repo

def _latest_sha(owner: str, repo: str, branch: str = "main") -> str:
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
        headers=HEADERS_GH,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["sha"]

def _parent_sha(owner: str, repo: str, sha: str) -> str:
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}",
        headers=HEADERS_GH,
        timeout=30,
    )
    r.raise_for_status()
    parents = r.json()["parents"]
    return parents[0]["sha"] if parents else sha

def get_latest_diff(max_lines: int = DIFF_LINES) -> str:
    if not (GH_URL and GH_PAT):
        return ""
    owner, repo = _owner_repo(GH_URL)
    head = _latest_sha(owner, repo)
    base = _parent_sha(owner, repo, head)
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
    r = requests.get(url, headers=HEADERS_GH, timeout=60)
    r.raise_for_status()
    diff = r.text.splitlines()[:max_lines]
    return "\n".join(diff)

def _build_code_context(hits: List[Dict], n: int = 5) -> str:
    blocks = []
    for h in hits[:n]:
        header = f"File: {h['file']}\nClass: {h.get('clazz')}  Method: {h.get('method')}\n"
        source = h.get("source_code", "")
        blocks.append(header + source)
    return "\n---\n".join(blocks)

def answer_query(query: str) -> str:
    q_vec = embed([query])[0]
    hits = query_neo(q_vec)
    hits = rerank_with_cohere(query, hits)
    code_ctx = _build_code_context(hits)
    diff_ctx = get_latest_diff()
    messages = [
        {
            "role": "system",
            "content": CHAT_SYSTEM_PROMPT.format(code_context=code_ctx, diff=diff_ctx),
        },
        {"role": "user", "content": query},
    ]
    print(messages)
    resp = openai.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages)
    return resp.choices[0].message.content.strip()

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

if __name__ == "__main__":
    print("Type plain query to list hits, or 'a ' prefix to answer via LLM.")
    try:
        while True:
            try:
                raw = input("\n> ")
            except EOFError:
                break
            if not raw.strip():
                continue
            if raw.startswith("a "):
                print("\nAnswer:\n")
                print(textwrap.fill(answer_query(raw[2:].strip()), width=100))
                continue
            q = raw.strip()
            vec = embed([q])[0]
            res = rerank_with_cohere(q, query_neo(vec))
            for i, h in enumerate(res, 1):
                print(f"#{i:2d} {h['score']:.4f} {h['file']} {h.get('clazz','')} {h.get('method','')}")
    finally:
        driver.close()
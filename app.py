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
from datetime import datetime, timedelta

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
GH_URL = "https://github.com/adit-bala/agentic-rca"
GH_PAT = os.getenv("GH_REPO_INDEXER_TOKEN")

CHAT_SYSTEM_PROMPT = (
    "You are a code search and context gathering specialist. Your goal is to find and provide detailed context about specific code patterns, methods, or classes in the codebase.\n\n"
    "You will be given:\n"
    "1. <code_context> - Relevant code snippets from the repository, including:\n"
    "   - Method and class definitions\n"
    "   - File paths and locations\n"
    "   - Dependencies and references\n"
    "2. <recent_commits> - Recent commits to the repository\n\n"
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
    "3. Additional Context\n"
    "\n"
    "<code_context>{code_context}</code_context>\n\n"
    "<recent_commits>{recent_commits}</recent_commits>"
    "<hits>{hits}</hits>"
)

HEADERS_GH = {
    "Authorization": f"Bearer {GH_PAT}" if GH_PAT else None,
    "Accept": "application/vnd.github.v3+json",  # Default to JSON
}
HEADERS_GH = {k: v for k, v in HEADERS_GH.items() if v is not None}

# Separate headers for diff endpoints
DIFF_HEADERS_GH = {
    "Authorization": f"Bearer {GH_PAT}" if GH_PAT else None,
    "Accept": "application/vnd.github.v3.diff",
}
DIFF_HEADERS_GH = {k: v for k, v in DIFF_HEADERS_GH.items() if v is not None}

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
    if r.status_code != 200:
        print(f"Error getting latest SHA: {r.status_code}")
        print(f"Response: {r.text}")
        return ""
    try:
        return r.json()["sha"]
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response: {r.text}")
        return ""

def _parent_sha(owner: str, repo: str, sha: str) -> str:
    if not sha:
        return ""
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}",
        headers=HEADERS_GH,
        timeout=30,
    )
    if r.status_code != 200:
        print(f"Error getting parent SHA: {r.status_code}")
        print(f"Response: {r.text}")
        return sha
    try:
        parents = r.json()["parents"]
        return parents[0]["sha"] if parents else sha
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response: {r.text}")
        return sha

def _build_code_context(hits: List[Dict], n: int = 5) -> str:
    blocks = []
    for h in hits[:n]:
        header = f"File: {h['file']}\nClass: {h.get('clazz')}  Method: {h.get('method')}\n"
        source = h.get("source_code", "")
        blocks.append(header + source)
    return "\n---\n".join(blocks)

def get_recent_commits(hours) -> List[Dict]:
    if not (GH_URL and GH_PAT):
        print("GitHub URL or PAT not configured")
        return []
    
    owner, repo = _owner_repo(GH_URL)
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {
        "since": since,
        "per_page": 5  # Increased to show more commits
    }
    
    r = requests.get(url, headers=HEADERS_GH, params=params, timeout=30)
    if r.status_code != 200:
        print(f"Error getting recent commits: {r.status_code}")
        print(f"Response: {r.text}")
        return []
    
    try:
        commits = r.json()
    except Exception as e:
        print(f"Error parsing commits JSON: {e}")
        print(f"Response: {r.text}")
        return []
    
    commit_details = []
    for commit in commits:
        sha = commit["sha"]
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        commit_r = requests.get(commit_url, headers=HEADERS_GH, timeout=30)
        if commit_r.status_code != 200:
            print(f"Error getting commit details for {sha}: {commit_r.status_code}")
            continue
        
        try:
            commit_data = commit_r.json()
        except Exception as e:
            print(f"Error parsing commit details JSON: {e}")
            continue
        
        # Get the diff for this commit
        diff_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        diff_r = requests.get(diff_url, headers=DIFF_HEADERS_GH, timeout=30)
        if diff_r.status_code != 200:
            print(f"Error getting diff for {sha}: {diff_r.status_code}")
            continue
        
        commit_details.append({
            "sha": sha,
            "message": commit_data["commit"]["message"],
            "author": commit_data["commit"]["author"]["name"],
            "date": commit_data["commit"]["author"]["date"],
            "diff": diff_r.text  # Removed the DIFF_LINES limit to show full diff
        })
    
    return commit_details

def _build_recent_commits_context(commits: List[Dict]) -> str:
    if not commits:
        return "No recent commits found."
    
    blocks = []
    for commit in commits:
        block = f"""Commit: {commit['sha']}
Author: {commit['author']}
Date: {commit['date']}
Message: {commit['message']}

{'='*80}"""
        blocks.append(block)
    
    return "\n".join(blocks)

def answer_query(query: str, use_model: bool = False) -> str:
    q_vec = embed([query])[0]
    hits = query_neo(q_vec)
    hits = rerank_with_cohere(query, hits)
    code_ctx = _build_code_context(hits)
    recent_commits = get_recent_commits(hours=24)
    commits_ctx = _build_recent_commits_context(recent_commits)

    print(f"code_ctx: {code_ctx}")
    print(f"commits_ctx: {commits_ctx}")
    
    messages = [
        {
            "role": "system",
            "content": CHAT_SYSTEM_PROMPT.format(
                hits=hits,
                code_context=code_ctx,
                recent_commits=commits_ctx
            ),
        },
        {"role": "user", "content": query},
    ]
    if use_model:
        resp = openai.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages)
        return resp.choices[0].message.content.strip()
    return f"relevant hits: {hits}\n\nrelevant code snippets: {code_ctx}\n\nrecent commits: {commits_ctx}"

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
                print(textwrap.fill(answer_query(raw[2:].strip(), use_model=True), width=100))
                continue
            q = raw.strip()
            vec = embed([q])[0]
            res = rerank_with_cohere(q, query_neo(vec))
            for i, h in enumerate(res, 1):
                print(f"#{i:2d} {h['score']:.4f} {h['file']} {h.get('clazz','')} {h.get('method','')}")
    finally:
        driver.close()
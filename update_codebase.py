from __future__ import annotations
import json, os, sys, time, base64, requests, re
from datetime import datetime
from pathlib import Path
import preprocessing
import create_tables
import pandas as pd
from io import StringIO
from typing import Dict, List, Tuple

# env vars
TOKEN = os.getenv("GH_REPO_INDEXER_TOKEN")
if not TOKEN:
    sys.exit("GH_REPO_INDEXER_TOKEN not set")
BRANCH = os.getenv("BRANCH", "main")
INTERVAL = int(os.getenv("INTERVAL_SEC", "600"))
STATE_FILE = "latest_commit.json"
API_ROOT = "https://api.github.com"
HDRS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "repo-indexer/1.0",
}


def parse_repo_url(url: str):
    if url.startswith("git@"):
        path = url.split(":", 1)[1]
    elif url.startswith("https://"):
        path = re.sub(r"https://github.com/", "", url)
    else:
        raise ValueError("Unsupported GitHub URL")
    owner, repo = path.replace(".git", "").split("/", 1)
    return owner, repo


def latest_commit(owner: str, repo: str, branch: str):
    r = requests.get(f"{API_ROOT}/repos/{owner}/{repo}/commits", headers=HDRS, params={"sha": branch, "per_page": 1}, timeout=30)
    r.raise_for_status()
    return r.json()[0]["sha"]


def compare_commits(owner: str, repo: str, base: str, head: str):
    url = f"{API_ROOT}/repos/{owner}/{repo}/compare/{base}...{head}"
    r = requests.get(url, headers=HDRS, timeout=60)
    r.raise_for_status()
    return [(f["status"][0].upper(), f["filename"]) for f in r.json().get("files", [])]


def file_content(owner: str, repo: str, path: str, ref: str):
    url = f"{API_ROOT}/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, headers=HDRS, params={"ref": ref}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode(errors="replace")
    return data.get("content")


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("latest_commit")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(sha: str):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"latest_commit": sha}, f)


def process_changed_files(changed_files: List[Tuple[str, str]], owner: str, repo: str, head_sha: str) -> Tuple[List[dict], List[dict]]:
    """Process changed files in memory and return class and method data."""
    # Create in-memory file system
    files_by_language = {}
    
    # Download and organize files by language
    for status, path in changed_files:
        if status == "D":
            continue
            
        content = file_content(owner, repo, path, head_sha)
        if content is None:
            continue
            
        ext = Path(path).suffix
        if lang := preprocessing.get_language_from_extension(ext):
            if lang not in files_by_language:
                files_by_language[lang] = []
            files_by_language[lang].append((path, content))
    
    if not files_by_language:
        return [], []
    
    # Process files using the reusable method
    return preprocessing.process_codebase_in_memory(files_by_language)


def update_embeddings(class_data: List[dict], method_data: List[dict], repo_path: str):
    """Update embeddings for the changed files."""
    # Create DataFrames
    methods_df = pd.DataFrame(method_data) if method_data else pd.DataFrame(columns=["file_path", "class_name", "name", "doc_comment", "source_code", "references"])
    classes_df = pd.DataFrame(class_data) if class_data else pd.DataFrame(columns=["file_path", "class_name", "constructor_declaration", "method_declarations", "source_code", "references"])
    
    if methods_df.empty and classes_df.empty:
        print("No data to embed - skipping embedding update")
        return
    
    # Prepare text for embedding
    if not methods_df.empty:
        print(f"Embedding {len(methods_df)} method rows...")
        methods_df["text"] = methods_df.apply(
            lambda r: f"File: {r.file_path}\nClass: {r.class_name}\nMethod: {r.name}\n\n"
                      f"Source Code: {create_tables.clip_tokens(r.source_code, create_tables.MAX_TOKENS)}",
            axis=1
        )
        methods_df["embedding"] = create_tables.embed_batch(methods_df["text"].tolist())
    
    if not classes_df.empty:
        print(f"Embedding {len(classes_df)} class rows...")
        classes_df["text"] = classes_df.apply(
            lambda r: f"File: {r.file_path}\nClass: {r.class_name}\n\n"
                      f"Source Code: {create_tables.clip_tokens(r.source_code, create_tables.MAX_TOKENS)}",
            axis=1
        )
        classes_df["embedding"] = create_tables.embed_batch(classes_df["text"].tolist())
    
    # Update Neo4j
    driver = create_tables.GraphDatabase.driver(
        create_tables.NEO4J_URI,
        auth=(create_tables.NEO4J_USER, create_tables.NEO4J_PASS)
    )
    
    try:
        with driver.session() as sess:
            # Ensure indexes exist
            sess.execute_write(create_tables.ensure_vector_index, "Method")
            sess.execute_write(create_tables.ensure_vector_index, "Class")
            
            # Delete existing nodes for these files
            for file_path in methods_df["file_path"].unique():
                sess.run("MATCH (n) WHERE n.file_path = $path DETACH DELETE n", path=file_path)
            
            # Insert new nodes
            CHUNK = 500
            if not methods_df.empty:
                for start in range(0, len(methods_df), CHUNK):
                    chunk = methods_df.iloc[start:start+CHUNK]
                    sess.execute_write(create_tables.bulk_insert, "Method", chunk.to_dict("records"))
            if not classes_df.empty:
                for start in range(0, len(classes_df), CHUNK):
                    chunk = classes_df.iloc[start:start+CHUNK]
                    sess.execute_write(create_tables.bulk_insert, "Class", chunk.to_dict("records"))
    finally:
        driver.close()


def monitor(repo_url: str):
    owner, repo = parse_repo_url(repo_url)
    print(f"▶ Monitoring {owner}/{repo} ({BRANCH}) every {INTERVAL}s")
    print("Press Ctrl+C to stop monitoring")
    
    while True:
        try:
            head_sha = latest_commit(owner, repo, BRANCH)
            last_sha = load_state()
            
            if last_sha != head_sha:
                if last_sha:
                    changes = compare_commits(owner, repo, last_sha, head_sha)
                    if changes:
                        print(f"\nFound {len(changes)} changed files:")
                        for status, path in changes:
                            print(f"  {status}: {path}")
                        
                        class_data, method_data = process_changed_files(changes, owner, repo, head_sha)
                        if class_data or method_data:
                            print(f"\nProcessing {len(class_data)} classes and {len(method_data)} methods...")
                            update_embeddings(class_data, method_data, repo)
                            print("✓ Database updated successfully")
                        else:
                            print("No processable files found in changes")
                save_state(head_sha)
            else:
                print("\rNo changes – up to date", end="", flush=True)
            
            print(f"\nNext check in {INTERVAL}s ({datetime.now().isoformat()})")
            time.sleep(INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            print(f"Retrying in {INTERVAL}s...")
            time.sleep(INTERVAL)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: monitor_repo_api.py <github_repo_url>")
    monitor(sys.argv[1])

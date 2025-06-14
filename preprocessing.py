import os
import sys
from treesitter import Treesitter, LanguageEnum
from collections import defaultdict
import csv
from typing import List, Dict
from tree_sitter import Node
# tree_sitter_languages import removed - now using individual language packages
import subprocess
from pathlib import Path
import pathspec
# TODO: Add TypeScript support later
# Define your BLACKLIST_DIR, WHITELIST_FILES, NODE_TYPES, and REFERENCE_IDENTIFIERS here
BLACKLIST_DIR = [
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    ".idea",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".vscode",
    ".github",
    ".gitlab",
    ".angular",
    "cdk.out",
    ".aws-sam",
    ".terraform"
]
WHITELIST_FILES = [".java", ".py", ".js", ".rs", ".go"]
BLACKLIST_FILES = ["docker-compose.yml"]

NODE_TYPES = {
    "python": {
        "class": "class_definition",
        "method": "function_definition"
    },
    "java": {
        "class": "class_declaration",
        "method": "method_declaration"
    },
    "rust": {
        "class": "struct_item",
        "method": "function_item"
    },
    "javascript": {
        "class": "class_declaration",
        "method": "method_definition"
    },
    "go": {
        "class": "type_declaration",
        "method": "method_declaration"
    }
}

REFERENCE_IDENTIFIERS = {
    "python": {
        "class": "identifier",
        "method": "call",
        "child_field_name": "function"
    },
    "java": {
        "class": "identifier",
        "method": "method_invocation",
        "child_field_name": "name"
    },
    "javascript": {
        "class": "identifier",
        "method": "call_expression",
        "child_field_name": "function"
    },
    "rust": {
        "class": "identifier",
        "method": "call_expression",
        "child_field_name": "function"
    },
    "go": {
        "class": "type_identifier",
        "method": "call_expression",
        "child_field_name": "function"
    }
}

def build_spec(root: Path) -> pathspec.PathSpec:
    """Compile *all* .gitignore patterns under `root` into one matcher."""
    patterns = []
    for gi in root.rglob(".gitignore"):
        if "venv" in gi.parts:
            continue
        patterns.extend(gi.read_text().splitlines())
    print("patterns", patterns)
    return pathspec.GitIgnoreSpec.from_lines(patterns)

def get_language_from_extension(file_ext):
    FILE_EXTENSION_LANGUAGE_MAP = {
        ".java": LanguageEnum.JAVA,
        ".py": LanguageEnum.PYTHON,
        ".js": LanguageEnum.JAVASCRIPT,
        ".rs": LanguageEnum.RUST,
        ".go": LanguageEnum.GO
    }
    return FILE_EXTENSION_LANGUAGE_MAP.get(file_ext)

# TODO: this is not working as expected, this combines all .gitignore files into one spec whereas we should have the sub-specs for each .gitignore file
def load_files(codebase_path):
    root = Path(codebase_path).resolve()
    spec = build_spec(root)
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if spec.match_file(rel):
            continue
        if path.is_file():
            ext = path.suffix
            if ext in WHITELIST_FILES and path.name not in BLACKLIST_FILES:
                if (lang := get_language_from_extension(ext)):
                    print("adding", path)
                    files.append((path, lang))
    return files

def parse_code_files(file_list):
    class_data = []
    method_data = []

    all_class_names = set()
    all_method_names = set()

    files_by_language = defaultdict(list)
    for file_path, language in file_list:
        files_by_language[language].append(file_path)

    for language, files in files_by_language.items():
        treesitter_parser = Treesitter.create_treesitter(language)
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()
                file_bytes = code.encode()
                class_nodes, method_nodes = treesitter_parser.parse(file_bytes)

                # Process class nodes
                for class_node in class_nodes:
                    class_name = class_node.name
                    all_class_names.add(class_name)
                    class_data.append({
                        "file_path": file_path,
                        "class_name": class_name,
                        "constructor_declaration": "",  # Extract if needed
                        "method_declarations": "\n-----\n".join(class_node.method_declarations) if class_node.method_declarations else "",
                        "source_code": class_node.source_code,
                        "references": []  # Will populate later
                    })

                # Process method nodes
                for method_node in method_nodes:
                    method_name = method_node.name
                    all_method_names.add(method_name)
                    method_data.append({
                        "file_path": file_path,
                        "class_name": method_node.class_name if method_node.class_name else "",
                        "name": method_name,
                        "doc_comment": method_node.doc_comment,
                        "source_code": method_node.method_source_code,
                        "references": []  # Will populate later
                    })

    return class_data, method_data, all_class_names, all_method_names

def find_references(file_list, class_names, method_names):
    references = {'class': defaultdict(list), 'method': defaultdict(list)}
    files_by_language = defaultdict(list)
    
    # Convert names to sets for O(1) lookup
    class_names = set(class_names)
    method_names = set(method_names)

    for file_path, language in file_list:
        files_by_language[language].append(file_path)

    for language, files in files_by_language.items():
        treesitter_parser = Treesitter.create_treesitter(language)
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()
                file_bytes = code.encode()
                tree = treesitter_parser.parser.parse(file_bytes)
                
                # Single pass through the AST
                stack = [(tree.root_node, None)]
                while stack:
                    node, parent = stack.pop()
                    
                    # Check for identifiers
                    if node.type == 'identifier':
                        name = node.text.decode()
                        
                        # Check if it's a class reference
                        if name in class_names and parent and parent.type in ['type', 'class_type', 'object_creation_expression']:
                            references['class'][name].append({
                                "file": file_path,
                                "line": node.start_point[0] + 1,
                                "column": node.start_point[1] + 1,
                                "text": parent.text.decode()
                            })
                        
                        # Check if it's a method reference
                        if name in method_names and parent and parent.type in ['call_expression', 'method_invocation']:
                            references['method'][name].append({
                                "file": file_path,
                                "line": node.start_point[0] + 1,
                                "column": node.start_point[1] + 1,
                                "text": parent.text.decode()
                            })
                    
                    # Add children to stack with their parent
                    stack.extend((child, node) for child in node.children)

    return references

def create_output_directory(codebase_path):
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))
    codebase_folder_name = os.path.basename(normalized_path)
    output_directory = os.path.join("processed", codebase_folder_name)
    os.makedirs(output_directory, exist_ok=True)
    return output_directory

def write_class_data_to_csv(class_data, output_directory):
    output_file = os.path.join(output_directory, "class_data.csv")
    fieldnames = ["file_path", "class_name", "constructor_declaration", "method_declarations", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in class_data:
            references = row.get("references", [])
            row["references"] = "; ".join([f"{ref['file']}:{ref['line']}:{ref['column']}" for ref in references])
            writer.writerow(row)
    print(f"Class data written to {output_file}")

def write_method_data_to_csv(method_data, output_directory):
    output_file = os.path.join(output_directory, "method_data.csv")
    fieldnames = ["file_path", "class_name", "name", "doc_comment", "source_code", "references"]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_data:
            references = row.get("references", [])
            row["references"] = "; ".join([f"{ref['file']}:{ref['line']}:{ref['column']}" for ref in references])
            writer.writerow(row)
    print(f"Method data written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the codebase path as an argument.")
        sys.exit(1)
    codebase_path = sys.argv[1]

    files = load_files(codebase_path)
    class_data, method_data, class_names, method_names = parse_code_files(files)

    # Find references
    references = find_references(files, class_names, method_names)

    # Map references back to class and method data
    class_data_dict = {cd['class_name']: cd for cd in class_data}
    method_data_dict = {(md['class_name'], md['name']): md for md in method_data}

    for class_name, refs in references['class'].items():
        if class_name in class_data_dict:
            class_data_dict[class_name]['references'] = refs

    for method_name, refs in references['method'].items():
        # Find all methods with this name (since methods might have the same name in different classes)
        for key in method_data_dict:
            if key[1] == method_name:
                method_data_dict[key]['references'] = refs

    # Convert dictionaries back to lists
    class_data = list(class_data_dict.values())
    method_data = list(method_data_dict.values())

    output_directory = create_output_directory(codebase_path)
    write_class_data_to_csv(class_data, output_directory)
    write_method_data_to_csv(method_data, output_directory)

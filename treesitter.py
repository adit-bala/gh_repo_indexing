from abc import ABC
from tree_sitter import Language, Parser
from enum import Enum
import logging
import inspect

# TODO: Add TypeScript support later
# Map of language names to their tree-sitter modules
LANGUAGE_MODULES = {
    'python': None,
    'java': None,
    'rust': None,
    'javascript': None,
    'go': None
}

# Dynamically import language modules
for lang in LANGUAGE_MODULES:
    try:
        LANGUAGE_MODULES[lang] = __import__(f'tree_sitter_{lang}')
    except ImportError:
        logging.warning(f"Could not import tree-sitter-{lang}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LanguageEnum(Enum):
    JAVA = "java"
    PYTHON = "python"
    RUST = "rust"
    JAVASCRIPT = "javascript"
    GO = "go"
    UNKNOWN = "unknown"

LANGUAGE_QUERIES = {
    LanguageEnum.JAVA: {
        'class_query': """
            (class_declaration
                name: (identifier) @class.name)
        """,
        'method_query': """
            (method_declaration
                name: (identifier) @method.name)
        """,
        'doc_query': """
            (comment) @comment
        """
    },
    LanguageEnum.PYTHON: {
        'class_query': """
            (class_definition
                name: (identifier) @class.name)
        """,
        'method_query': """
            (function_definition
                name: (identifier) @method.name)
        """,
        'doc_query': """
            (string) @comment
        """
    },
    LanguageEnum.RUST: {
        'class_query': """
            (struct_item
                name: (type_identifier) @class.name)
        """,
        'method_query': """
            (function_item
                name: (identifier) @function.name)
        """,
        'doc_query': """
            [
                (line_comment) @comment
                (block_comment) @comment
            ]
        """
    },
    LanguageEnum.JAVASCRIPT: {
        'class_query': """
            (class_declaration
                name: (identifier) @class.name)
        """,
        'method_query': """
            (method_definition
                name: (property_identifier) @method.name)
        """,
        'doc_query': """
            [
                (comment) @comment
                (jsx_text) @comment
            ]
        """
    },
    LanguageEnum.GO: {
        'class_query': """
            (type_declaration
                (type_spec
                    name: (type_identifier) @class.name
                    type: (struct_type)))
        """,
        'method_query': """
            (method_declaration
                name: (field_identifier) @method.name)
        """,
        'doc_query': """
            [
                (comment) @comment
            ]
        """
    }
}

class TreesitterMethodNode:
    def __init__(
        self,
        name: str,
        doc_comment: str,
        method_source_code: str,
        node,
        class_name: str = None
    ):
        self.name = name
        self.doc_comment = doc_comment
        self.method_source_code = method_source_code
        self.node = node
        self.class_name = class_name

class TreesitterClassNode:
    def __init__(
        self,
        name: str,
        method_declarations: list,
        node,
    ):
        self.name = name
        self.source_code = node.text.decode()
        self.method_declarations = method_declarations
        self.node = node

class Treesitter(ABC):
    def __init__(self, language: LanguageEnum):
        self.language_enum = language
        self.parser, self.language_obj = self._get_parser_and_language(language.value)
        self.query_config = LANGUAGE_QUERIES.get(language)
        if not self.query_config:
            raise ValueError(f"Unsupported language: {language}")

        # Corrected query instantiation
        self.class_query = self.language_obj.query(self.query_config['class_query'])
        self.method_query = self.language_obj.query(self.query_config['method_query'])
        self.doc_query = self.language_obj.query(self.query_config['doc_query'])

    @staticmethod
    def create_treesitter(language: LanguageEnum) -> "Treesitter":
        return Treesitter(language)

    def parse(self, file_bytes: bytes) -> tuple[list[TreesitterClassNode], list[TreesitterMethodNode]]:
        tree = self.parser.parse(file_bytes)
        root_node = tree.root_node

        class_results = []
        method_results = []

        class_name_by_node = {}
        class_captures = self.class_query.captures(root_node)
        class_nodes = []
        
        # New API returns a dict with capture names as keys
        if 'class.name' in class_captures:
            for node in class_captures['class.name']:
                class_name = node.text.decode()
                class_node = node.parent
                logging.info(f"Found class: {class_name}")
                class_name_by_node[class_node.id] = class_name
                method_declarations = self._extract_methods_in_class(class_node)
                class_results.append(TreesitterClassNode(class_name, method_declarations, class_node))
                class_nodes.append(class_node)

        method_captures = self.method_query.captures(root_node)
        for capture_name in ['method.name', 'function.name']:
            if capture_name in method_captures:
                for node in method_captures[capture_name]:
                    method_name = node.text.decode()
                    method_node = node.parent
                    method_source_code = method_node.text.decode()
                    doc_comment = self._extract_doc_comment(method_node)
                    parent_class_name = None
                    for class_node in class_nodes:
                        if self._is_descendant_of(method_node, class_node):
                            parent_class_name = class_name_by_node[class_node.id]
                            break
                    method_results.append(TreesitterMethodNode(
                        name=method_name,
                        doc_comment=doc_comment,
                        method_source_code=method_source_code,
                        node=method_node,
                        class_name=parent_class_name
                    ))

        return class_results, method_results

    def _extract_methods_in_class(self, class_node):
        method_declarations = []
        # Apply method_query to the class_node
        method_captures = self.method_query.captures(class_node)
        for capture_name in ['method.name', 'function.name']:
            if capture_name in method_captures:
                for node in method_captures[capture_name]:
                    method_declaration = node.parent.text.decode()
                    method_declarations.append(method_declaration)
        return method_declarations

    def _extract_doc_comment(self, node):
        # Search for doc comments preceding the node
        doc_comment = ''
        current_node = node.prev_sibling
        while current_node:
            captures = self.doc_query.captures(current_node)
            if captures and 'comment' in captures:
                for cap_node in captures['comment']:
                    doc_comment = cap_node.text.decode() + '\n' + doc_comment
            elif current_node.type not in ['comment', 'block_comment', 'line_comment', 'expression_statement']:
                # Stop if we reach a node that's not a comment
                break
            current_node = current_node.prev_sibling
        return doc_comment.strip()

    def _is_descendant_of(self, node, ancestor):
        # Check if 'node' is a descendant of 'ancestor'
        current = node.parent
        while current:
            if current == ancestor:
                return True
            current = current.parent
        return False
    
    def _get_parser_and_language(self, language_name: str):
        """Get parser and language object for the specified language."""
        module = LANGUAGE_MODULES.get(language_name)
        if not module:
            raise ValueError(f"Unsupported language: {language_name}")
        if language_name == "typescript":
            language_capsule = LANGUAGE_MODULES.get("javascript").language()
        else:
            language_capsule = module.language()
        language_obj = Language(language_capsule)
        parser = Parser(language_obj)
        return parser, language_obj

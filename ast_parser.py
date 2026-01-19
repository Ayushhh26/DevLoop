"""
Multi-Language AST Parser

Extracts complete function and class definitions from source code
using tree-sitter for multi-language support.

Supported languages:
- Python
- JavaScript / TypeScript
- Java
- C / C++
- Go
- Rust
- Ruby
- C#
"""

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import importlib.util


@dataclass
class CodeBlock:
    """Represents a complete code block (function, class, or method)."""
    name: str
    type: str  # 'function', 'class', 'method', 'interface', etc.
    start_line: int
    end_line: int
    source: str
    language: str
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    signature: Optional[str] = None


# Check if tree-sitter is available
TREE_SITTER_AVAILABLE = importlib.util.find_spec("tree_sitter") is not None

if TREE_SITTER_AVAILABLE:
    try:
        import tree_sitter
        from tree_sitter import Language, Parser
        
        # Language module imports (may not all be available)
        LANGUAGE_MODULES = {}
        
        language_imports = [
            ('python', 'tree_sitter_python'),
            ('javascript', 'tree_sitter_javascript'),
            ('typescript', 'tree_sitter_typescript'),
            ('java', 'tree_sitter_java'),
            ('c', 'tree_sitter_c'),
            ('cpp', 'tree_sitter_cpp'),
            ('go', 'tree_sitter_go'),
            ('rust', 'tree_sitter_rust'),
            ('ruby', 'tree_sitter_ruby'),
            ('c_sharp', 'tree_sitter_c_sharp'),
        ]
        
        for lang_name, module_name in language_imports:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'language'):
                    # Wrap the PyCapsule in Language() for newer tree-sitter versions
                    raw_lang = module.language()
                    LANGUAGE_MODULES[lang_name] = Language(raw_lang)
            except (ImportError, TypeError) as e:
                # If Language() wrapper fails, try direct assignment (older versions)
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, 'language'):
                        LANGUAGE_MODULES[lang_name] = module.language()
                except ImportError:
                    pass
                
    except ImportError:
        TREE_SITTER_AVAILABLE = False
        LANGUAGE_MODULES = {}


class TreeSitterParser:
    """
    Multi-language parser using tree-sitter.
    
    Extracts complete functions, classes, methods, and other code blocks
    from source code in various programming languages.
    """
    
    # Node types that represent function/method definitions per language
    FUNCTION_NODES = {
        'python': ['function_definition', 'async_function_definition'],
        'javascript': ['function_declaration', 'arrow_function', 'method_definition', 'function_expression'],
        'typescript': ['function_declaration', 'arrow_function', 'method_definition', 'function_expression'],
        'java': ['method_declaration', 'constructor_declaration'],
        'c': ['function_definition'],
        'cpp': ['function_definition', 'template_declaration'],
        'go': ['function_declaration', 'method_declaration'],
        'rust': ['function_item', 'impl_item'],
        'ruby': ['method', 'singleton_method'],
        'c_sharp': ['method_declaration', 'constructor_declaration'],
    }
    
    # Node types that represent class/struct definitions
    CLASS_NODES = {
        'python': ['class_definition'],
        'javascript': ['class_declaration', 'class_expression'],
        'typescript': ['class_declaration', 'interface_declaration', 'type_alias_declaration'],
        'java': ['class_declaration', 'interface_declaration', 'enum_declaration'],
        'c': ['struct_specifier', 'union_specifier', 'enum_specifier'],
        'cpp': ['class_specifier', 'struct_specifier', 'union_specifier', 'enum_specifier'],
        'go': ['type_declaration'],
        'rust': ['struct_item', 'enum_item', 'trait_item', 'impl_item'],
        'ruby': ['class', 'module'],
        'c_sharp': ['class_declaration', 'interface_declaration', 'struct_declaration'],
    }
    
    def __init__(self, source_code: str, language: str):
        self.source_code = source_code
        self.source_bytes = source_code.encode('utf-8')
        self.lines = source_code.splitlines(keepends=True)
        self.language = language
        self.code_blocks: List[CodeBlock] = []
        self.parser = None
        self.tree = None
        
        if TREE_SITTER_AVAILABLE and language in LANGUAGE_MODULES:
            self.parser = Parser()
            self.parser.language = LANGUAGE_MODULES[language]
            
    def parse(self) -> List[CodeBlock]:
        """Parse the source code and extract all code blocks."""
        if self.parser is None:
            # Fall back to Python AST for Python files
            if self.language == 'python':
                return self._parse_python_ast()
            return []
        
        try:
            self.tree = self.parser.parse(self.source_bytes)
            self._extract_blocks(self.tree.root_node)
        except Exception as e:
            # If tree-sitter fails, try Python AST fallback
            if self.language == 'python':
                return self._parse_python_ast()
            print(f"Warning: tree-sitter parsing failed for {self.language}: {e}")
        
        return self.code_blocks
    
    def _parse_python_ast(self) -> List[CodeBlock]:
        """Fallback parser using Python's built-in AST module."""
        try:
            tree = ast.parse(self.source_code)
            self._extract_python_blocks(tree)
        except SyntaxError:
            pass
        return self.code_blocks
    
    def _extract_python_blocks(self, tree: ast.AST, parent_class: Optional[str] = None):
        """Extract blocks from Python AST."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                start_line = node.lineno
                if node.decorator_list:
                    start_line = min(d.lineno for d in node.decorator_list)
                end_line = self._get_python_node_end(node)
                source = self._get_source_segment(start_line, end_line)
                
                block = CodeBlock(
                    name=node.name,
                    type='class',
                    start_line=start_line,
                    end_line=end_line,
                    source=source,
                    language='python',
                    docstring=ast.get_docstring(node),
                    parent_class=parent_class,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                )
                self.code_blocks.append(block)
                self._extract_python_blocks(node, parent_class=node.name)
                
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_async = isinstance(node, ast.AsyncFunctionDef)
                block_type = 'async_method' if parent_class and is_async else \
                            'method' if parent_class else \
                            'async_function' if is_async else 'function'
                
                start_line = node.lineno
                if node.decorator_list:
                    start_line = min(d.lineno for d in node.decorator_list)
                end_line = self._get_python_node_end(node)
                source = self._get_source_segment(start_line, end_line)
                
                block = CodeBlock(
                    name=node.name,
                    type=block_type,
                    start_line=start_line,
                    end_line=end_line,
                    source=source,
                    language='python',
                    docstring=ast.get_docstring(node),
                    parent_class=parent_class,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                )
                self.code_blocks.append(block)
    
    def _get_decorator_name(self, dec: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(dec, ast.Name):
            return dec.id
        elif isinstance(dec, ast.Attribute):
            return ast.unparse(dec)
        elif isinstance(dec, ast.Call):
            return self._get_decorator_name(dec.func)
        return str(dec)
    
    def _get_python_node_end(self, node: ast.AST) -> int:
        """Get end line of Python AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        if hasattr(node, 'body') and node.body:
            return self._get_python_node_end(node.body[-1])
        return node.lineno
    
    def _get_source_segment(self, start_line: int, end_line: int) -> str:
        """Extract source code between line numbers (1-indexed)."""
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self.lines), end_line)
        return ''.join(self.lines[start_idx:end_idx])
    
    def _extract_blocks(self, node, parent_class: Optional[str] = None):
        """Recursively extract code blocks from tree-sitter AST."""
        node_type = node.type
        
        # Check if this is a function node
        func_nodes = self.FUNCTION_NODES.get(self.language, [])
        class_nodes = self.CLASS_NODES.get(self.language, [])
        
        if node_type in func_nodes:
            block = self._create_block_from_node(node, 'function', parent_class)
            if block:
                self.code_blocks.append(block)
                
        elif node_type in class_nodes:
            block = self._create_block_from_node(node, 'class', parent_class)
            if block:
                self.code_blocks.append(block)
                # Recursively extract methods from class body
                class_name = block.name
                for child in node.children:
                    self._extract_blocks(child, parent_class=class_name)
                return  # Don't double-process children
        
        # Recurse into children
        for child in node.children:
            self._extract_blocks(child, parent_class)
    
    def _create_block_from_node(self, node, block_type: str, parent_class: Optional[str]) -> Optional[CodeBlock]:
        """Create a CodeBlock from a tree-sitter node."""
        try:
            start_line = node.start_point[0] + 1  # tree-sitter is 0-indexed
            end_line = node.end_point[0] + 1
            
            # Get the full source text
            source = self.source_code[node.start_byte:node.end_byte]
            
            # Try to extract the name
            name = self._extract_name(node)
            if not name:
                name = f"anonymous_{block_type}_{start_line}"
            
            # Determine more specific type
            if parent_class and block_type == 'function':
                block_type = 'method'
            
            return CodeBlock(
                name=name,
                type=block_type,
                start_line=start_line,
                end_line=end_line,
                source=source,
                language=self.language,
                parent_class=parent_class,
            )
        except Exception as e:
            print(f"Warning: Could not create block from node: {e}")
            return None
    
    def _extract_name(self, node) -> Optional[str]:
        """Extract the name from a tree-sitter node."""
        # Look for identifier child nodes
        for child in node.children:
            if child.type in ('identifier', 'name', 'property_identifier', 
                            'type_identifier', 'field_identifier'):
                return self.source_code[child.start_byte:child.end_byte]
            
            # For some languages, name might be nested
            if child.type in ('declarator', 'function_declarator'):
                return self._extract_name(child)
        
        return None


def extract_functions(source_code: str, language: str) -> List[CodeBlock]:
    """
    Extract all functions and classes from source code.
    
    Args:
        source_code: Source code string
        language: Programming language ('python', 'javascript', 'java', etc.)
        
    Returns:
        List of CodeBlock objects representing functions/classes
    """
    parser = TreeSitterParser(source_code, language)
    return parser.parse()


def extract_functions_from_python(source_code: str) -> List[CodeBlock]:
    """
    Legacy function for backward compatibility.
    Extract all functions and classes from Python source code.
    """
    return extract_functions(source_code, 'python')


def find_function_by_name(source_code: str, function_name: str, language: str = 'python') -> Optional[CodeBlock]:
    """
    Find a specific function/method by name in source code.
    """
    blocks = extract_functions(source_code, language)
    for block in blocks:
        if block.name == function_name:
            return block
    return None


def get_supported_languages() -> List[str]:
    """Return list of supported languages."""
    base_languages = ['python']  # Always supported via AST fallback
    
    if TREE_SITTER_AVAILABLE:
        base_languages.extend(LANGUAGE_MODULES.keys())
    
    return list(set(base_languages))


# Language to file extension mapping
LANGUAGE_EXTENSIONS = {
    '.py': 'python',
    '.pyw': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.mjs': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.cs': 'c_sharp',
}


def get_language_from_extension(extension: str) -> Optional[str]:
    """Get language name from file extension."""
    return LANGUAGE_EXTENSIONS.get(extension.lower())


if __name__ == "__main__":
    # Test the parser
    print(f"Tree-sitter available: {TREE_SITTER_AVAILABLE}")
    print(f"Available languages: {get_supported_languages()}")
    
    # Test Python parsing
    test_python = '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    async def fetch_and_add(self, url: str) -> int:
        """Fetch a number and add it."""
        pass


def standalone_function(x: int) -> int:
    """A standalone function."""
    return x * 2
'''
    
    print("\n--- Python Parsing ---")
    blocks = extract_functions(test_python, 'python')
    for block in blocks:
        print(f"{block.type}: {block.name} (lines {block.start_line}-{block.end_line})")
    
    # Test JavaScript if available
    if 'javascript' in get_supported_languages():
        test_js = '''
class UserService {
    constructor(db) {
        this.db = db;
    }
    
    async getUser(id) {
        return await this.db.find(id);
    }
}

function calculateTotal(items) {
    return items.reduce((sum, item) => sum + item.price, 0);
}

const arrowFunc = (x, y) => x + y;
'''
        print("\n--- JavaScript Parsing ---")
        blocks = extract_functions(test_js, 'javascript')
        for block in blocks:
            print(f"{block.type}: {block.name} (lines {block.start_line}-{block.end_line})")

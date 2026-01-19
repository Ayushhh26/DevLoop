"""
RAG Ingestion Engine

This module handles repository cloning, code parsing, and vector storage.
Uses AST-based extraction for Python to get complete function bodies.
Falls back to language-aware splitting for other languages.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import git
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Import AST parser for full function extraction (multi-language)
from ast_parser import extract_functions, CodeBlock, LANGUAGE_EXTENSIONS, get_supported_languages


def sanitize_url(url: str) -> str:
    """Clean and sanitize a GitHub URL."""
    # Strip whitespace
    url = url.strip()
    # Remove any invisible/special characters that might get copied
    url = ''.join(char for char in url if ord(char) < 128 or char.isalnum() or char in '/:.-_')
    
    # Fix common typos like https://ttps:// or https://https://
    url = re.sub(r'^https?://+h?t?t?p?s?:?/?/?', 'https://', url)
    
    # Ensure proper https:// prefix
    if url.startswith('http://'):
        url = url.replace('http://', 'https://', 1)
    elif not url.startswith('https://'):
        if 'github.com' in url:
            url = 'https://' + url.lstrip('/')
        else:
            url = f'https://github.com/{url}'
    
    # Remove any double slashes (except after https:)
    url = re.sub(r'(?<!:)//+', '/', url)
    
    return url


def get_repo_name(repo_url: str) -> str:
    """Extract repository name from GitHub URL."""
    # Handle various URL formats: https://github.com/user/repo, https://github.com/user/repo.git, user/repo
    match = re.search(r'(?:github\.com[/:])?([^/]+)/([^/]+?)(?:\.git)?/?$', repo_url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    raise ValueError(f"Invalid repository URL format: {repo_url}")


def read_code_files(repo_path: Path) -> List[Tuple[str, str, str]]:
    """
    Recursively read code files from repository.
    Returns list of (file_path, content, language) tuples.
    
    Supports: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, C#
    """
    code_files = []
    
    # Comprehensive extension mapping for all supported languages
    extensions = {
        # Python
        '.py': 'python',
        '.pyw': 'python',
        # JavaScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.mjs': 'javascript',
        # TypeScript
        '.ts': 'typescript',
        '.tsx': 'typescript',
        # Java
        '.java': 'java',
        # C
        '.c': 'c',
        '.h': 'c',
        # C++
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        '.hh': 'cpp',
        # Go
        '.go': 'go',
        # Rust
        '.rs': 'rust',
        # Ruby
        '.rb': 'ruby',
        # C#
        '.cs': 'c_sharp',
        # Additional web files (text chunking only)
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
    }
    
    # Directories to skip
    skip_dirs = {
        'node_modules', '__pycache__', 'venv', 'env', '.venv',
        'dist', 'build', 'target', 'bin', 'obj',
        '.git', '.svn', '.hg', '.idea', '.vscode',
        'vendor', 'packages', '.next', '.nuxt',
    }
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]
        
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            if ext in extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Skip empty or very large files
                        if not content.strip() or len(content) > 500000:  # Skip files > 500KB
                            continue
                        # Get relative path from repo root
                        rel_path = file_path.relative_to(repo_path)
                        code_files.append((str(rel_path), content, extensions[ext]))
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue
    
    return code_files


def split_code_by_language(text: str, language: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split code using language-aware separators.
    Uses RecursiveCharacterTextSplitter with language-specific separators to preserve structure.
    """
    # Language-specific separators (in order of priority)
    separators_map = {
        'python': [
            '\n\n\n',  # Classes/functions separated by multiple newlines
            '\n\n',    # Function/class definitions
            '\n',      # Single newlines
            ' ',       # Spaces
            '',        # Characters
        ],
        'javascript': [
            '\n\n\n',
            '\n\n',
            '\n',
            ' ',
            '',
        ],
        'typescript': [
            '\n\n\n',
            '\n\n',
            '\n',
            ' ',
            '',
        ],
    }
    
    separators = separators_map.get(language, ['\n\n', '\n', ' ', ''])
    
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return splitter.split_text(text)


def extract_code_functions(content: str, file_path: str, language: str, repo_name: str, max_chunk_size: int = 2000) -> Optional[List[Document]]:
    """
    Extract complete functions/classes from source code using AST.
    
    Supports multiple languages via tree-sitter:
    - Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, C#
    
    Falls back to regular chunking for:
    - Very large functions
    - Unsupported languages
    - Parse failures
    
    Args:
        content: Source code string
        file_path: Path to the file (for metadata)
        language: Programming language
        repo_name: Repository name (for metadata)
        max_chunk_size: Maximum size before splitting a function
        
    Returns:
        List of Document objects with full function metadata, or None if extraction fails
    """
    # Languages that support AST extraction
    ast_supported = {'python', 'javascript', 'typescript', 'java', 'c', 'cpp', 'go', 'rust', 'ruby', 'c_sharp'}
    
    if language not in ast_supported:
        return None
    
    documents = []
    
    try:
        code_blocks = extract_functions(content, language)
    except Exception as e:
        print(f"Warning: AST extraction failed for {file_path}: {e}")
        return None
    
    if not code_blocks:
        # AST parsing failed or no functions found
        return None
    
    # Track which lines are covered by extracted blocks
    covered_lines = set()
    
    for block in code_blocks:
        # Skip methods - they're already included in their class
        # But DO include standalone functions and classes
        if block.type in ('method', 'async_method'):
            continue
            
        source = block.source
        
        # If function is too large, split it but keep metadata
        if len(source) > max_chunk_size:
            # Split large functions but preserve metadata
            chunks = split_code_by_language(source, language, chunk_size=max_chunk_size, chunk_overlap=200)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'file_path': file_path,
                        'language': language,
                        'source': repo_name,
                        'block_type': block.type,
                        'block_name': block.name,
                        'start_line': block.start_line,
                        'end_line': block.end_line,
                        'is_complete': False,  # Indicates this is a partial chunk
                        'chunk_part': idx + 1,
                        'parent_class': block.parent_class,
                        'docstring': block.docstring[:200] if block.docstring else None,
                    }
                )
                documents.append(doc)
        else:
            # Store complete function as a single document
            doc = Document(
                page_content=source,
                metadata={
                    'file_path': file_path,
                    'language': language,
                    'source': repo_name,
                    'block_type': block.type,
                    'block_name': block.name,
                    'start_line': block.start_line,
                    'end_line': block.end_line,
                    'is_complete': True,  # This is a complete function
                    'parent_class': block.parent_class,
                    'docstring': block.docstring[:200] if block.docstring else None,
                }
            )
            documents.append(doc)
        
        # Mark these lines as covered
        for line in range(block.start_line, block.end_line + 1):
            covered_lines.add(line)
    
    # Also add file-level code (imports, constants, etc.) that's not in any function
    lines = content.splitlines(keepends=True)
    uncovered_content = []
    current_start = None
    
    for i, line in enumerate(lines, 1):
        if i not in covered_lines:
            if current_start is None:
                current_start = i
            uncovered_content.append(line)
        else:
            if uncovered_content and len(''.join(uncovered_content).strip()) > 50:
                # Store uncovered content (imports, module-level code)
                doc = Document(
                    page_content=''.join(uncovered_content),
                    metadata={
                        'file_path': file_path,
                        'language': language,
                        'source': repo_name,
                        'block_type': 'module_level',
                        'block_name': 'imports_and_constants',
                        'start_line': current_start,
                        'end_line': i - 1,
                        'is_complete': True,
                    }
                )
                documents.append(doc)
            uncovered_content = []
            current_start = None
    
    # Handle any remaining uncovered content
    if uncovered_content and len(''.join(uncovered_content).strip()) > 50:
        doc = Document(
            page_content=''.join(uncovered_content),
            metadata={
                'file_path': file_path,
                'language': language,
                'source': repo_name,
                'block_type': 'module_level',
                'block_name': 'imports_and_constants',
                'start_line': current_start,
                'end_line': len(lines),
                'is_complete': True,
            }
        )
        documents.append(doc)
    
    return documents if documents else None


def ingest_repo(repo_url: str) -> str:
    """
    Ingest a GitHub repository: clone, parse code files, create embeddings, and store in ChromaDB.
    
    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/user/repo)
    
    Returns:
        Status message with ingestion count
    """
    try:
        # Sanitize URL to handle copy-paste issues
        repo_url = sanitize_url(repo_url)
        print(f"Sanitized URL: {repo_url}")
        
        # Get repository name
        repo_name = get_repo_name(repo_url)
        repo_path = Path("data") / repo_name
        
        # Clone repository (or update if exists)
        if repo_path.exists():
            print(f"Repository {repo_name} already exists. Updating...")
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            print(f"Cloning repository {repo_url}...")
            repo = git.Repo.clone_from(repo_url, repo_path)
        
        # Read code files
        print("Reading code files...")
        code_files = read_code_files(repo_path)
        
        if not code_files:
            return f"Error: No code files found in repository {repo_url}"
        
        # Initialize embeddings (CPU-based, free and private)
        print("Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Process files and create documents
        print("Processing and chunking code files...")
        print(f"Supported AST languages: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, C#")
        
        documents = []
        total_chunks = 0
        ast_extracted = 0
        language_stats = {}
        
        for file_path, content, language in code_files:
            # Track language statistics
            language_stats[language] = language_stats.get(language, 0) + 1
            
            # Try AST-based extraction first (works for multiple languages)
            ast_docs = extract_code_functions(content, file_path, language, repo_name)
            if ast_docs:
                documents.extend(ast_docs)
                total_chunks += len(ast_docs)
                ast_extracted += len(ast_docs)
                continue  # Successfully extracted with AST
            
            # Fall back to regular chunking if AST extraction failed or unsupported language
            chunks = split_code_by_language(content, language, chunk_size=1000, chunk_overlap=200)
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'file_path': file_path,
                        'language': language,
                        'chunk_index': idx,
                        'source': repo_name,
                        'is_complete': False,  # Regular chunking, may be incomplete
                    }
                )
                documents.append(doc)
                total_chunks += 1
        
        # Print statistics
        print(f"Language breakdown: {language_stats}")
        print(f"AST-extracted {ast_extracted} complete functions/classes.")
        
        # Store in ChromaDB (persistent storage)
        print(f"Storing {total_chunks} chunks in ChromaDB...")
        persist_directory = "./chroma_db"
        
        # Create or load vectorstore
        # Use repo_name as collection name to allow multiple repos
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=repo_name,
        )
        
        # Persist to disk
        vectorstore.persist()
        
        return f"Successfully ingested repository '{repo_name}'. Processed {len(code_files)} files into {total_chunks} chunks."
    
    except git.exc.GitCommandError as e:
        return f"Error cloning repository: {str(e)}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error during ingestion: {str(e)}"


if __name__ == "__main__":
    # Example usage
    test_url = "https://github.com/example/test-repo"
    result = ingest_repo(test_url)
    print(result)

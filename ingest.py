"""
RAG Ingestion Engine

This module handles repository cloning, code parsing, and vector storage.
Uses AST-based extraction for Python to get complete function bodies.
Falls back to language-aware splitting for other languages.
"""

import os
import re
import signal
import sys
import subprocess
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from contextlib import contextmanager

import git
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Import AST parser for full function extraction (multi-language)
from ast_parser import extract_functions, CodeBlock, LANGUAGE_EXTENSIONS, get_supported_languages

# Configuration constants
MAX_REPO_SIZE_MB = 500  # Maximum repository size in MB
MAX_FILES_TO_PROCESS = 10000  # Maximum number of files to process
GIT_CLONE_TIMEOUT = 300  # 5 minutes timeout for git clone
GIT_PULL_TIMEOUT = 60  # 1 minute timeout for git pull
MAX_FILE_SIZE_BYTES = 500000  # 500KB per file (already implemented)
MAX_TOTAL_CHUNKS = 50000  # Maximum total chunks to prevent memory exhaustion
MAX_TOTAL_CONTENT_MB = 100  # Maximum total content size to process (MB)
BATCH_SIZE = 100  # Process files in batches to manage memory


def validate_github_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate that URL is a legitimate GitHub URL (prevent SSRF attacks).
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"
    
    # Check length limit
    if len(url) > 2000:
        return False, "URL is too long (max 2000 characters)"
    
    # Must be https://github.com or https://www.github.com
    github_pattern = r'^https://(www\.)?github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(/.*)?$'
    
    # Prevent SSRF: reject localhost, file://, etc.
    dangerous_patterns = [
        r'localhost',
        r'127\.0\.0\.1',
        r'0\.0\.0\.0',
        r'file://',
        r'ftp://',
        r'://[^/]*\.local',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return False, f"URL contains potentially unsafe pattern: {pattern}"
    
    if not re.match(github_pattern, url):
        return False, "Invalid GitHub URL format. Must be https://github.com/owner/repo"
    
    return True, None


def sanitize_url(url: str) -> str:
    """
    Clean and sanitize a GitHub URL.
    
    Args:
        url: Raw URL string
        
    Returns:
        Sanitized URL
        
    Raises:
        ValueError: If URL is invalid or not a GitHub URL
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    
    # Check length limit (prevent extremely long URLs)
    if len(url) > 500:
        raise ValueError("URL is too long (max 500 characters)")
    
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
    
    # Validate it's actually a GitHub URL (prevent SSRF)
    is_valid, error_msg = validate_github_url(url)
    if not is_valid:
        raise ValueError(f"Invalid GitHub URL: {error_msg}")
    
    return url


def get_repo_name(repo_url: str) -> str:
    """Extract repository name from GitHub URL."""
    # Handle various URL formats: https://github.com/user/repo, https://github.com/user/repo.git, user/repo
    match = re.search(r'(?:github\.com[/:])?([^/]+)/([^/]+?)(?:\.git)?/?$', repo_url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    raise ValueError(f"Invalid repository URL format: {repo_url}")


def clone_repo_with_timeout(repo_url: str, repo_path: Path, timeout: int = GIT_CLONE_TIMEOUT) -> git.Repo:
    """
    Clone repository with timeout protection.
    
    Args:
        repo_url: GitHub repository URL
        repo_path: Path where to clone the repository
        timeout: Timeout in seconds
        
    Returns:
        Git Repo object
        
    Raises:
        TimeoutError: If clone operation exceeds timeout
        git.exc.GitCommandError: If git command fails
    """
    def clone_operation():
        return git.Repo.clone_from(repo_url, repo_path)
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = clone_operation()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        # Try to clean up the partial clone
        if repo_path.exists():
            import shutil
            try:
                shutil.rmtree(repo_path)
            except:
                pass
        raise TimeoutError(f"Git clone operation timed out after {timeout} seconds")
    
    if exception[0]:
        raise exception[0]
    
    if result[0] is None:
        raise Exception("Git clone operation failed without raising an exception")
    
    return result[0]


def pull_repo_with_timeout(repo: git.Repo, timeout: int = GIT_PULL_TIMEOUT) -> None:
    """
    Pull repository updates with timeout protection.
    
    Args:
        repo: Git Repo object
        timeout: Timeout in seconds
        
    Raises:
        TimeoutError: If pull operation exceeds timeout
        git.exc.GitCommandError: If git command fails
    """
    def pull_operation():
        repo.remotes.origin.pull()
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = pull_operation()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Git pull operation timed out after {timeout} seconds")
    
    if exception[0]:
        raise exception[0]
    
    if result[0] is None:
        raise Exception("Git pull operation failed without raising an exception")


def check_repo_size_estimate(repo_url: str) -> int:
    """
    Estimate repository size using git ls-remote (doesn't require full clone).
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Estimated size in bytes (0 if unable to determine)
    """
    try:
        # Use git ls-remote to get a rough estimate
        # This is a lightweight operation that doesn't clone
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', '--tags', repo_url],
            capture_output=True,
            timeout=30,
            text=True
        )
        if result.returncode == 0:
            # Rough estimate: count refs and multiply by average size
            # This is a heuristic - actual size may vary significantly
            ref_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            # Very rough estimate: assume 1-5MB per ref (this is conservative)
            estimated_bytes = ref_count * 3 * 1024 * 1024  # 3MB per ref
            return estimated_bytes
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
        # If we can't estimate, return 0 (will skip size check)
        pass
    return 0


def read_code_files(repo_path: Path, max_files: int = MAX_FILES_TO_PROCESS, progress_callback: Optional[Callable] = None) -> List[Tuple[str, str, str]]:
    """
    Recursively read code files from repository with memory limits.
    Returns list of (file_path, content, language) tuples.
    
    Supports: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, C#
    
    Args:
        repo_path: Path to repository root
        max_files: Maximum number of files to process
        
    Returns:
        List of (file_path, content, language) tuples
    """
    code_files = []
    total_content_size = 0
    max_total_size_bytes = MAX_TOTAL_CONTENT_MB * 1024 * 1024
    should_stop = False  # Flag to break out of nested loops
    
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
        # Check if we should stop (from previous iteration)
        if should_stop:
            break
        
        # Check file count limit before processing this directory
        if len(code_files) >= max_files:
            print(f"Warning: Reached maximum file limit ({max_files}). Stopping file processing.")
            should_stop = True
            break
        
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]
        
        for file in files:
            # Check if we should stop (from previous iteration)
            if should_stop:
                break
            
            # Check file count limit again inside loop
            if len(code_files) >= max_files:
                print(f"Warning: Reached maximum file limit ({max_files}). Stopping file processing.")
                should_stop = True
                break
            
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            if ext in extensions:
                
                try:
                    # Check file size before reading
                    file_size = file_path.stat().st_size
                    if file_size > MAX_FILE_SIZE_BYTES:
                        continue  # Skip files > 500KB
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Skip empty files
                        if not content.strip():
                            continue
                        
                        # Check total content size limit
                        content_size = len(content.encode('utf-8'))
                        if total_content_size + content_size > max_total_size_bytes:
                            print(f"Warning: Reached maximum content size limit ({MAX_TOTAL_CONTENT_MB}MB). Stopping file processing.")
                            should_stop = True
                            break
                        
                        # Check file count limit again after size check
                        if len(code_files) >= max_files:
                            print(f"Warning: Reached maximum file limit ({max_files}). Stopping file processing.")
                            should_stop = True
                            break
                        
                        total_content_size += content_size
                        # Get relative path from repo root
                        rel_path = file_path.relative_to(repo_path)
                        code_files.append((str(rel_path), content, extensions[ext]))
                        
                        # Update progress callback if provided (every 50 files to avoid spam)
                        if progress_callback and len(code_files) % 50 == 0:
                            progress_callback(len(code_files), max_files, str(rel_path))
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


def ingest_repo(repo_url: str, progress_callback: Optional[Callable] = None) -> str:
    """
    Ingest a GitHub repository: clone, parse code files, create embeddings, and store in ChromaDB.
    
    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/user/repo)
        progress_callback: Optional callback function for progress updates
                          (receives stage, progress, status_message)
    
    Returns:
        Status message with ingestion count
    """
    try:
        # Helper function to update progress
        def update_progress(stage_name: str, progress: float, status: str = None):
            if progress_callback:
                progress_callback(stage_name, progress, status)
            else:
                print(f"[{int(progress * 100)}%] {status or stage_name}")
        
        # Stage 1: Validating URL (0-5%)
        update_progress("validating", 0.02, "Validating URL...")
        repo_url = sanitize_url(repo_url)
        
        # Stage 2: Checking repository size (5-10%)
        update_progress("checking_size", 0.05, "Checking repository size...")
        repo_name = get_repo_name(repo_url)
        repo_path = Path("data") / repo_name
        
        estimated_size = check_repo_size_estimate(repo_url)
        if estimated_size > 0:
            estimated_size_mb = estimated_size / (1024 * 1024)
            if estimated_size_mb > MAX_REPO_SIZE_MB:
                if progress_callback:
                    progress_callback("error", 0.0, f"Repository size ({estimated_size_mb:.1f}MB) exceeds maximum ({MAX_REPO_SIZE_MB}MB)")
                return f"Error: Repository size ({estimated_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_REPO_SIZE_MB}MB)"
            update_progress("checking_size", 0.10, f"Repository size: {estimated_size_mb:.1f}MB")
        
        # Stage 3: Clone/Pull repository (10-30%)
        if repo_path.exists():
            update_progress("pulling", 0.15, f"Repository exists. Updating {repo_name}...")
            repo = git.Repo(repo_path)
            pull_repo_with_timeout(repo, timeout=GIT_PULL_TIMEOUT)
            update_progress("pulling", 0.30, "Repository updated")
        else:
            update_progress("cloning", 0.15, f"Cloning repository {repo_name}...")
            repo = clone_repo_with_timeout(repo_url, repo_path, timeout=GIT_CLONE_TIMEOUT)
            update_progress("cloning", 0.30, "Repository cloned")
        
        # Stage 4: Read code files (30-50%)
        update_progress("reading_files", 0.35, "Reading code files...")
        def file_progress_callback(current, total, filename=None):
            if total > 0:
                progress = 0.35 + min((current / total) * 0.15, 0.15)
                status = f"Reading files: {current}/{total}"
                if filename:
                    status += f" - {filename[:40]}..."
                update_progress("reading_files", progress, status)
        code_files = read_code_files(repo_path, progress_callback=file_progress_callback)
        
        if not code_files:
            if progress_callback:
                progress_callback("error", 0.0, "No code files found")
            return f"Error: No code files found in repository {repo_url}"
        
        update_progress("reading_files", 0.50, f"Found {len(code_files)} code files")
        
        # Stage 5: Initialize embeddings (50-55%)
        update_progress("initializing", 0.50, "Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        update_progress("initializing", 0.55, "Embeddings model ready")
        
        # Stage 6: Process files and create documents (55-85%)
        update_progress("processing", 0.55, f"Processing {len(code_files)} files...")
        
        documents = []
        total_chunks = 0
        ast_extracted = 0
        language_stats = {}
        
        # Process files in batches to manage memory
        total_batches = (len(code_files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx, batch_start in enumerate(range(0, len(code_files), BATCH_SIZE)):
            batch_end = min(batch_start + BATCH_SIZE, len(code_files))
            batch = code_files[batch_start:batch_end]
            
            # Update progress for batch processing
            batch_progress = 0.55 + (batch_idx / total_batches) * 0.25
            update_progress("processing", batch_progress, 
                          f"Processing batch {batch_idx + 1}/{total_batches} ({batch_start + 1}-{batch_end} of {len(code_files)})...")
            
            for file_idx, (file_path, content, language) in enumerate(batch):
                # Update file-level progress
                file_progress = batch_progress + (file_idx / len(batch)) * (0.25 / total_batches)
                if progress_callback and file_idx % 10 == 0:  # Update every 10 files to avoid spam
                    update_progress("processing", file_progress, 
                                  f"Processing: {file_path[:50]}...")
                # Check chunk limit
                if total_chunks >= MAX_TOTAL_CHUNKS:
                    print(f"Warning: Reached maximum chunk limit ({MAX_TOTAL_CHUNKS}). Stopping processing.")
                    break
                
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
                    if total_chunks >= MAX_TOTAL_CHUNKS:
                        break
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
            
            if total_chunks >= MAX_TOTAL_CHUNKS:
                break
        
        # Stage 7: Creating embeddings (85-95%)
        update_progress("creating_embeddings", 0.85, f"Creating embeddings for {total_chunks} chunks...")
        
        # Print statistics (for logging)
        if not progress_callback:
            print(f"Language breakdown: {language_stats}")
            print(f"AST-extracted {ast_extracted} complete functions/classes.")
        
        # Stage 8: Store in ChromaDB (95-100%)
        update_progress("storing", 0.95, f"Storing {total_chunks} chunks in vector database...")
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
        
        update_progress("complete", 1.0, f"Complete! Processed {len(code_files)} files into {total_chunks} chunks")
        
        return f"Successfully ingested repository '{repo_name}'. Processed {len(code_files)} files into {total_chunks} chunks."
    
    except TimeoutError as e:
        error_msg = str(e)
        if "clone" in error_msg.lower():
            return f"Error: Git clone operation timed out after {GIT_CLONE_TIMEOUT} seconds. The repository may be too large or the network connection is slow. Please try again or use a smaller repository."
        elif "pull" in error_msg.lower():
            return f"Error: Git pull operation timed out after {GIT_PULL_TIMEOUT} seconds. Please try again or re-clone the repository."
        else:
            return f"Error: Operation timed out. {error_msg}"
    except git.exc.GitCommandError as e:
        error_str = str(e).lower()
        if "not found" in error_str or "does not exist" in error_str:
            return f"Error: Repository not found. Please check the URL and ensure the repository is public or you have access."
        elif "authentication" in error_str or "permission" in error_str:
            return f"Error: Authentication failed. Please ensure the repository is public or check your credentials."
        else:
            return f"Error cloning repository: {str(e)}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except MemoryError as e:
        return f"Error: Insufficient memory to process this repository. The repository may be too large. Please try a smaller repository or increase available memory."
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        return f"Unexpected error during ingestion ({error_type}): {str(e)}"


if __name__ == "__main__":
    # Example usage
    test_url = "https://github.com/example/test-repo"
    result = ingest_repo(test_url)
    print(result)

"""
RAG Ingestion Engine

This module handles repository cloning, code parsing, and vector storage.
Uses language-aware code splitting to preserve function/class boundaries.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

import git
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def sanitize_url(url: str) -> str:
    """Clean and sanitize a GitHub URL."""
    # Strip whitespace
    url = url.strip()
    # Remove any invisible/special characters that might get copied
    url = ''.join(char for char in url if ord(char) < 128 or char.isalnum())
    # Ensure proper https:// prefix
    if url.startswith('http://'):
        url = url.replace('http://', 'https://', 1)
    elif not url.startswith('https://'):
        if 'github.com' in url:
            url = 'https://' + url.lstrip('/')
        else:
            url = f'https://github.com/{url}'
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
    """
    code_files = []
    extensions = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
    }
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
        
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            if ext in extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
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
        documents = []
        total_chunks = 0
        
        for file_path, content, language in code_files:
            # Split code using language-aware chunking
            chunks = split_code_by_language(content, language, chunk_size=1000, chunk_overlap=200)
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'file_path': file_path,
                        'language': language,
                        'chunk_index': idx,
                        'source': repo_name,
                    }
                )
                documents.append(doc)
                total_chunks += 1
        
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

"""
Repo-Chat: Multi-Agent RAG Code Analysis System

Streamlit UI for repository ingestion and interactive code analysis.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import ingest
import agent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Repo-Chat",
    page_icon="💬",
    layout="wide",
)

# Initialize session state
if "ingested_repo_url" not in st.session_state:
    st.session_state.ingested_repo_url = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None


def get_repo_name_from_url(repo_url: str) -> Optional[str]:
    """Extract repository name from URL."""
    import re
    match = re.search(r'(?:github\.com[/:])?([^/]+)/([^/]+?)(?:\.git)?/?$', repo_url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None


def load_vectorstore(repo_name: str):
    """Load ChromaDB vectorstore for a repository."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        persist_directory = "./chroma_db"
        
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=repo_name,
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None


def extract_keywords(query: str) -> List[str]:
    """
    Extract important keywords from query for hybrid search.
    Focuses on function names, class names, and technical terms.
    """
    keywords = []
    
    # Pattern 1: "function_name function" or "the function_name function"
    pattern1 = r'\b([a-z_][a-z0-9_]*)\s+(?:function|method|class)'
    matches = re.findall(pattern1, query, re.IGNORECASE)
    keywords.extend(matches)
    
    # Pattern 2: "function_name()" 
    pattern2 = r'\b([a-z_][a-z0-9_]+)\s*\('
    matches = re.findall(pattern2, query, re.IGNORECASE)
    keywords.extend(matches)
    
    # Pattern 3: snake_case identifiers (likely function/variable names)
    pattern3 = r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b'
    matches = re.findall(pattern3, query, re.IGNORECASE)
    keywords.extend(matches)
    
    # Pattern 4: CamelCase identifiers (likely class names)
    pattern4 = r'\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+)\b'
    matches = re.findall(pattern4, query)
    keywords.extend(matches)
    
    # Remove common words and short keywords
    stop_words = {'the', 'and', 'are', 'not', 'with', 'for', 'this', 'that', 'from', 'have'}
    keywords = [kw for kw in keywords if kw.lower() not in stop_words and len(kw) > 3]
    
    return list(set(keywords))  # Remove duplicates


def keyword_search(vectorstore: Chroma, keywords: List[str], k: int = 20) -> List[dict]:
    """
    Perform keyword-based search by filtering documents that contain specific keywords.
    This complements semantic search by finding exact matches.
    """
    keyword_results = []
    
    try:
        # Access the underlying ChromaDB collection
        collection = vectorstore._collection
        
        for keyword in keywords:
            # Use ChromaDB's where_document filter for substring matching
            try:
                results = collection.get(
                    where_document={"$contains": keyword},
                    limit=k,
                    include=["documents", "metadatas"]
                )
                
                if results and results['documents']:
                    for i, doc_content in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if results['metadatas'] else {}
                        # Use a very low score (0.1) to prioritize keyword matches
                        keyword_results.append({
                            'content': doc_content,
                            'metadata': metadata,
                            'score': 0.1,  # Low score = high priority (lower is better)
                            'match_type': 'keyword',
                            'matched_keyword': keyword,
                        })
            except Exception as e:
                # If where_document filter fails, continue with other keywords
                continue
                
    except Exception as e:
        # If we can't access the collection, return empty results
        pass
    
    return keyword_results


def retrieve_context(vectorstore: Chroma, query: str, k: int = 15) -> List[dict]:
    """
    HYBRID SEARCH: Combines semantic search with keyword-based search.
    
    Strategy:
    1. Extract keywords (function names, class names) from query
    2. Perform keyword search (exact text matching)
    3. Perform semantic search (meaning-based matching)
    4. Merge results, prioritizing:
       - Keyword matches
       - Complete functions (AST-extracted)
       - Source files over test files
    5. Deduplicate by file path
    
    Returns:
        List of dictionaries with 'content' and 'metadata', sorted by relevance
    """
    try:
        # Step 1: Extract keywords from query
        keywords = extract_keywords(query)
        
        # Step 2: Keyword search (exact text matching)
        keyword_results = keyword_search(vectorstore, keywords, k=k * 2)
        
        # Step 3: Semantic search
        semantic_k = min(k * 3, 50)
        semantic_results = vectorstore.similarity_search_with_score(query, k=semantic_k)
        
        # Convert semantic results to same format
        semantic_list = []
        for doc, score in semantic_results:
            semantic_list.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score,
                'match_type': 'semantic',
            })
        
        # Step 4: Merge results
        # Keyword matches come first (lower score = higher priority)
        all_results = keyword_results + semantic_list
        
        # Step 4.5: Apply scoring adjustments
        for result in all_results:
            file_path = result['metadata'].get('file_path', '').lower()
            is_complete = result['metadata'].get('is_complete', False)
            block_name = result['metadata'].get('block_name', '')
            language = result['metadata'].get('language', '')
            
            # BOOST: Complete functions (AST-extracted) are more valuable
            if is_complete:
                result['score'] = result['score'] * 0.7  # Strong boost for complete functions
            
            # BOOST: If function name matches a keyword in query
            if block_name and any(kw.lower() in block_name.lower() for kw in keywords):
                result['score'] = result['score'] * 0.5  # Very strong boost for name match
            
            # HEAVILY PENALIZE: Documentation files (not source code)
            doc_patterns = ['changelog', 'readme', 'benchmark', 'docs/', 'doc/', '.md', 'license', 'contributing']
            is_doc = any(pattern in file_path for pattern in doc_patterns)
            
            # PENALIZE: Test files (they show usage, not implementation)
            test_patterns = ['test', 'spec', 'mock', '__test__', '_test.py', '_test.go', '_test.js', 'tests/']
            is_test = any(pattern in file_path for pattern in test_patterns)
            
            # Apply penalties/boosts
            if is_doc:
                result['score'] = result['score'] * 3.0  # Heavy penalty for docs
            elif is_test:
                result['score'] = result['score'] * 2.0  # Penalty for test files
            
            # BOOST: Source code files (actual implementation)
            source_patterns = ['src/', 'lib/', 'core/', 'app/', 'pkg/', 'internal/', 'cmd/']
            if any(pattern in file_path for pattern in source_patterns):
                result['score'] = result['score'] * 0.7  # Boost for source dirs
            
            # BOOST: Main source files at root level (common in Go, Python)
            root_source_extensions = ['.go', '.py', '.rs', '.java', '.js', '.ts']
            if '/' not in file_path and any(file_path.endswith(ext) for ext in root_source_extensions):
                result['score'] = result['score'] * 0.6  # Strong boost for root-level source files
        
        # Deduplicate by content (keep first occurrence, which has better score)
        seen_content = set()
        unique_results = []
        for result in all_results:
            content_key = result['content'][:200]  # Use first 200 chars as key
            if content_key not in seen_content:
                unique_results.append(result)
                seen_content.add(content_key)
        
        # Step 5: Sort and select best results
        # Sort by score (lower is better)
        unique_results.sort(key=lambda x: x['score'])
        
        # Prefer complete functions, but also include variety
        complete_functions = [r for r in unique_results if r['metadata'].get('is_complete', False)]
        partial_chunks = [r for r in unique_results if not r['metadata'].get('is_complete', False)]
        
        context_list = []
        seen_files = set()
        
        # First, add best complete functions (up to 70% of k)
        max_complete = int(k * 0.7)
        for result in complete_functions:
            if len(context_list) >= max_complete:
                break
            file_path = result['metadata'].get('file_path', 'unknown')
            block_name = result['metadata'].get('block_name', '')
            key = f"{file_path}:{block_name}"
            
            if key not in seen_files:
                context_list.append(result)
                seen_files.add(key)
        
        # Then, fill remaining slots with partial chunks for additional context
        for result in partial_chunks:
            if len(context_list) >= k:
                break
            file_path = result['metadata'].get('file_path', 'unknown')
            
            if file_path not in seen_files:
                context_list.append(result)
                seen_files.add(file_path)
        
        # If still need more, add additional results
        for result in unique_results:
            if len(context_list) >= k:
                break
            if result not in context_list:
                context_list.append(result)
        
        return context_list[:k]
        
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []


# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # GitHub URL input
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/user/repo",
        value=st.session_state.ingested_repo_url or "",
    )
    
    # LLM Provider selection
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["GROQ", "DEEPSEEK"],
        index=0 if os.getenv("LLM_PROVIDER", "GROQ") == "GROQ" else 1,
    )
    os.environ["LLM_PROVIDER"] = llm_provider
    
    # API Key status (read from .env file only)
    st.caption("API keys are loaded from `.env` file")
    
    # Ingest Repository button
    st.divider()
    if st.button("🔄 Ingest Repository", type="primary", use_container_width=True):
        if not repo_url:
            st.error("Please enter a GitHub repository URL")
        else:
            with st.spinner("Ingesting repository... This may take a few minutes."):
                try:
                    result = ingest.ingest_repo(repo_url)
                    st.success(result)
                    
                    # Update session state
                    repo_name = get_repo_name_from_url(repo_url)
                    if repo_name:
                        st.session_state.repo_name = repo_name
                        st.session_state.ingested_repo_url = repo_url
                        st.session_state.vectorstore = load_vectorstore(repo_name)
                        if st.session_state.vectorstore:
                            st.session_state.chat_history = []
                            st.rerun()
                except Exception as e:
                    st.error(f"Error ingesting repository: {e}")
    
    # Display current repository
    if st.session_state.ingested_repo_url:
        st.divider()
        st.info(f"📁 Current Repo:\n{st.session_state.ingested_repo_url}")
        if st.button("Clear Repository", use_container_width=True):
            st.session_state.ingested_repo_url = None
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.repo_name = None
            st.rerun()


# Main area
st.title("💬 Repo-Chat")
st.markdown("Ask questions about your codebase and get AI-powered fixes with quality assurance.")

# Check if repository is ingested
if st.session_state.vectorstore is None:
    st.info("👈 Please ingest a repository using the sidebar to get started.")
    st.markdown("""
    ### How to use:
    1. Set your API key in the `.env` file (e.g., `GROQ_API_KEY=gsk_...`)
    2. Enter a GitHub repository URL in the sidebar
    3. Click "Ingest Repository" to analyze the codebase
    4. Start chatting about your code!
    """)
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display retrieved files if available
            if message["role"] == "assistant" and "retrieved_files" in message:
                with st.expander("📄 Retrieved Code Files", expanded=False):
                    for file_info in message["retrieved_files"]:
                        st.markdown(f"**{file_info['file_path']}** (Score: {file_info['score']:.3f})")
                        st.code(file_info['content'], language=file_info.get('language', 'python'))
            
            # Display draft code if available
            if message["role"] == "assistant" and "draft_code" in message:
                st.markdown("#### 💻 Generated Code Fix:")
                st.code(message["draft_code"], language="python")
            
            # Display critic review
            if message["role"] == "assistant" and "critique" in message:
                status = message.get("final_status", "REJECT")
                if status == "APPROVE":
                    st.success(f"✅ **Critic Review: APPROVED**\n\n{message['critique']}")
                else:
                    st.error(f"❌ **Critic Review: REJECTED**\n\n{message['critique']}")
    
    # Chat input
    user_query = st.chat_input("Ask about your codebase or request a fix...")
    
    if user_query:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Validate API keys (must be set in .env file)
                    if llm_provider == "GROQ" and not os.getenv("GROQ_API_KEY"):
                        st.error("Groq API key is required. Please set `GROQ_API_KEY` in your `.env` file.")
                        st.stop()
                    elif llm_provider == "DEEPSEEK" and not os.getenv("DEEPSEEK_API_KEY"):
                        st.error("DeepSeek API key is required. Please set `DEEPSEEK_API_KEY` in your `.env` file.")
                        st.stop()
                    
                    # Retrieve relevant context (increased k for better retrieval)
                    retrieved_context_list = retrieve_context(st.session_state.vectorstore, user_query, k=10)
                    
                    if not retrieved_context_list:
                        st.warning("No relevant code found. Try rephrasing your query.")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I couldn't find relevant code for your query. Please try rephrasing.",
                        })
                        st.stop()
                    
                    # Display retrieved files
                    retrieved_files = []
                    context_texts = []
                    for ctx in retrieved_context_list:
                        block_name = ctx['metadata'].get('block_name', '')
                        block_type = ctx['metadata'].get('block_type', '')
                        is_complete = ctx['metadata'].get('is_complete', False)
                        
                        retrieved_files.append({
                            'file_path': ctx['metadata'].get('file_path', 'unknown'),
                            'content': ctx['content'],
                            'language': ctx['metadata'].get('language', 'python'),
                            'score': ctx['score'],
                            'block_name': block_name,
                            'block_type': block_type,
                            'is_complete': is_complete,
                        })
                        context_texts.append(ctx['content'])
                    
                    with st.expander("📄 Retrieved Code Files", expanded=True):
                        for file_info in retrieved_files:
                            # Build display header
                            header = f"**{file_info['file_path']}**"
                            if file_info['block_name']:
                                header += f" → `{file_info['block_name']}`"
                                if file_info['block_type']:
                                    header += f" ({file_info['block_type']})"
                            
                            # Show completeness badge
                            if file_info['is_complete']:
                                header += " ✅ Complete"
                            
                            st.markdown(header)
                            st.code(file_info['content'][:800] + "\n# ... (truncated)" if len(file_info['content']) > 800 else file_info['content'], 
                                   language=file_info.get('language', 'python'))
                    
                    # Run agent loop
                    st.markdown("🤖 Running agentic loop...")
                    result = agent.run_agent_loop(
                        query=user_query,
                        context=context_texts,
                        provider=llm_provider,
                        max_iterations=3,
                    )
                    
                    # Display draft code
                    st.markdown("#### 💻 Generated Code Fix:")
                    st.code(result["draft_code"], language="python")
                    
                    # Display critic review
                    status = result["final_status"]
                    if status == "APPROVE":
                        st.success(f"✅ **Critic Review: APPROVED** (Iterations: {result['iterations']})\n\n{result['critique']}")
                    else:
                        st.error(f"❌ **Critic Review: REJECTED** (Iterations: {result['iterations']})\n\n{result['critique']}")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Here's the analysis for: {user_query}",
                        "retrieved_files": retrieved_files,
                        "draft_code": result["draft_code"],
                        "critique": result["critique"],
                        "final_status": result["final_status"],
                    })
                
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    import traceback
                    st.code(traceback.format_exc())

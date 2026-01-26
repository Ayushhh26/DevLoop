"""
Progress Tracking Utility for Streamlit

Provides a unified interface for tracking progress across different operations
(ingestion, agent loops, retrieval, etc.)
"""

from typing import Optional, Callable
from enum import Enum


class ProgressStage(Enum):
    """Enumeration of progress stages"""
    VALIDATING = "Validating URL..."
    CHECKING_SIZE = "Checking repository size..."
    CLONING = "Cloning repository..."
    PULLING = "Updating repository..."
    READING_FILES = "Reading code files..."
    PROCESSING_FILES = "Processing files..."
    EXTRACTING_AST = "Extracting functions (AST)..."
    CHUNKING = "Chunking code..."
    CREATING_EMBEDDINGS = "Creating embeddings..."
    STORING = "Storing in vector database..."
    COMPLETE = "Complete"


class ProgressTracker:
    """
    Progress tracker that works with Streamlit UI.
    
    Usage:
        tracker = ProgressTracker(st.progress(0), st.empty())
        tracker.update_stage(ProgressStage.CLONING, 0.3)
        tracker.update_status("Cloning repository...", 0.35)
    """
    
    def __init__(self, progress_bar=None, status_text=None):
        """
        Initialize progress tracker.
        
        Args:
            progress_bar: Streamlit progress bar (st.progress)
            status_text: Streamlit empty container (st.empty()) for status text
        """
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.current_stage = None
        self.current_progress = 0.0
        
    def update_stage(self, stage: ProgressStage, progress: float, status: Optional[str] = None):
        """
        Update progress stage and percentage.
        
        Args:
            stage: Current progress stage
            progress: Progress percentage (0.0 to 1.0)
            status: Optional status message (if None, uses stage description)
        """
        self.current_stage = stage
        self.current_progress = max(0.0, min(1.0, progress))
        
        if self.progress_bar:
            self.progress_bar.progress(self.current_progress)
        
        status_msg = status or stage.value
        if self.status_text:
            self.status_text.text(f"{status_msg} ({int(self.current_progress * 100)}%)")
    
    def update_status(self, status: str, progress: Optional[float] = None):
        """
        Update status message and optionally progress.
        
        Args:
            status: Status message
            progress: Optional progress percentage (0.0 to 1.0)
        """
        if progress is not None:
            self.current_progress = max(0.0, min(1.0, progress))
            if self.progress_bar:
                self.progress_bar.progress(self.current_progress)
        
        if self.status_text:
            percent = int(self.current_progress * 100)
            self.status_text.text(f"{status} ({percent}%)")
    
    def update_file_progress(self, current: int, total: int, filename: Optional[str] = None):
        """
        Update file processing progress.
        
        Args:
            current: Current file number
            total: Total files
            filename: Optional filename being processed
        """
        progress = current / total if total > 0 else 0.0
        self.current_progress = progress
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text:
            if filename:
                # Truncate long filenames
                display_name = filename if len(filename) <= 50 else filename[:47] + "..."
                self.status_text.text(f"Processing file {current}/{total}: {display_name} ({int(progress * 100)}%)")
            else:
                self.status_text.text(f"Processing file {current}/{total} ({int(progress * 100)}%)")
    
    def update_chunk_progress(self, current: int, total: int):
        """
        Update chunk creation progress.
        
        Args:
            current: Current chunk number
            total: Total chunks (estimated or actual)
        """
        progress = current / total if total > 0 else 0.0
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text:
            self.status_text.text(f"Created {current}/{total} chunks ({int(progress * 100)}%)")
    
    def complete(self, message: str = "Complete!"):
        """
        Mark progress as complete.
        
        Args:
            message: Completion message
        """
        self.current_progress = 1.0
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            self.status_text.text(message)
    
    def error(self, error_message: str):
        """
        Mark progress as error.
        
        Args:
            error_message: Error message
        """
        if self.status_text:
            self.status_text.text(f"❌ Error: {error_message}")


class AgentProgressTracker:
    """
    Progress tracker specifically for agent loops.
    """
    
    def __init__(self, status_text=None, iteration_text=None):
        """
        Initialize agent progress tracker.
        
        Args:
            status_text: Streamlit empty container for status
            iteration_text: Streamlit empty container for iteration info
        """
        self.status_text = status_text
        self.iteration_text = iteration_text
        self.current_iteration = 0
        self.max_iterations = 3
    
    def update_iteration(self, iteration: int, max_iterations: int, agent: str):
        """
        Update iteration progress.
        
        Args:
            iteration: Current iteration (1-indexed)
            max_iterations: Maximum iterations
            agent: Current agent ("Coder" or "Critic")
        """
        self.current_iteration = iteration
        self.max_iterations = max_iterations
        
        if self.iteration_text:
            self.iteration_text.markdown(f"**Iteration {iteration}/{max_iterations}**")
        
        if self.status_text:
            agent_emoji = "💻" if agent == "Coder" else "🔍"
            self.status_text.text(f"{agent_emoji} {agent} Agent: {'Generating code...' if agent == 'Coder' else 'Reviewing code...'}")
    
    def update_status(self, status: str):
        """
        Update status message.
        
        Args:
            status: Status message
        """
        if self.status_text:
            self.status_text.text(status)
    
    def complete(self, approved: bool):
        """
        Mark agent loop as complete.
        
        Args:
            approved: Whether the code was approved
        """
        if self.status_text:
            status = "✅ Code approved!" if approved else "❌ Code rejected after review"
            self.status_text.text(status)

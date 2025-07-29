"""Pipeline management and Git integration utilities."""
import subprocess
import sys
import os
import tempfile
import threading
import time
import random
from pathlib import Path
from datetime import datetime
import streamlit as st
from loguru import logger


class PipelineChecker:
    """Handles pipeline file checking and execution prerequisites."""
    
    @staticmethod
    def check_pipeline_exists():
        """Check if pipeline.py exists and return its path."""
        # Get the actual current working directory
        current_dir = Path.cwd()
        
        # List of possible locations for pipeline.py
        possible_paths = [
            current_dir / "pipeline.py",  # Current directory (src)
            current_dir / "src" / "pipeline.py",  # In case we're in parent dir
            current_dir.parent / "src" / "pipeline.py",  # Parent/src structure
            Path(__file__).parent / "pipeline.py" if '__file__' in globals() else current_dir / "pipeline.py",
        ]
        
        # Debug: show where we're looking
        print(f"Current working directory: {current_dir}")
        print(f"Looking for pipeline.py in these locations:")
        for i, path in enumerate(possible_paths, 1):
            exists = path.exists()
            print(f"  {i}. {path} {'✓' if exists else '✗'}")
            if exists:
                return path, True
        
        return possible_paths[0], False  # Return first path as default, but indicate not found

    @staticmethod
    def should_run_pipeline(config):
        """Check if pipeline should run based on data freshness (3+ days)."""
        try:
            # First check if pipeline.py exists
            pipeline_path, exists = PipelineChecker.check_pipeline_exists()
            if not exists:
                st.warning(f"⚠️ pipeline.py not found at expected location: {pipeline_path}")
                st.info("Please make sure pipeline.py is in the same directory as your main script.")
                return False  # Don't try to run if file doesn't exist
            
            data_dir = Path(config.data_paths.get('processed', 'data/processed'))
            if not data_dir.exists():
                return True
                
            # Check age of newest file
            newest_file = max(data_dir.glob('*.csv'), key=os.path.getctime, default=None)
            if not newest_file:
                return True
                
            # Calculate file age in days
            file_age_days = (datetime.now().timestamp() - os.path.getctime(newest_file)) / 86400  # 86400 seconds in a day
            return file_age_days >= 3  # Run if data is 3 or more days old
            
        except Exception as e:
            logger.warning(f"Pipeline check failed: {e}")
            return True  # Run pipeline if check fails


class PipelineRunner:
    """Handles pipeline execution with progress tracking."""
    
    def __init__(self):
        self.progress_messages = [
            # 0-10%: Starting up
            "🚀 Initializing data pipeline...",
            "📡 Connecting to data sources...",
            "🔍 Scanning for available datasets...",
            
            # 10-30%: Data fetching
            "📊 Fetching energy consumption data...",
            "🌡️ Downloading weather information...",
            "📈 Retrieving historical trends...",
            "🔄 Synchronizing data streams...",
            
            # 30-50%: Processing
            "⚙️ Processing raw data files...",
            "🧹 Cleaning and validating datasets...",
            "🔗 Linking weather and energy data...",
            "📏 Normalizing data formats...",
            
            # 50-70%: Analysis
            "🔬 Analyzing consumption patterns...",
            "📊 Computing statistical correlations...",
            "🎯 Identifying peak usage periods...",
            "🌟 Detecting seasonal trends...",
            
            # 70-90%: Finalization
            "💾 Saving processed datasets...",
            "📋 Generating summary reports...",
            "🔧 Optimizing data structures...",
            "✨ Preparing visualizations...",
            
            # 90-95%: Final steps
            "🎨 Finalizing dashboard components...",
            "🔍 Running quality checks...",
            "📦 Packaging results..."
        ]
    
    def run_pipeline_with_progress(self):
        """Run pipeline with animated progress bar and engaging messages."""
        # Create progress bar and status placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Progress tracking
        progress_value = 0
        pipeline_complete = False
        pipeline_result = None
        message_index = 0
        
        def run_pipeline():
            nonlocal pipeline_complete, pipeline_result
            try:
                # Get the pipeline path
                pipeline_path, exists = PipelineChecker.check_pipeline_exists()
                
                if not exists:
                    raise FileNotFoundError(f"pipeline.py not found at any expected location")
                
                print(f"Found pipeline.py at: {pipeline_path}")
                
                # Set working directory to where pipeline.py is located
                working_dir = pipeline_path.parent
                print(f"Running pipeline from directory: {working_dir}")
                
                # Check if config directory exists in the working directory
                config_dir = working_dir / "config"
                if not config_dir.exists():
                    # Try to find config directory in parent or other locations
                    possible_config_dirs = [
                        working_dir.parent / "config",  # Parent directory
                        working_dir / ".." / "config",  # Relative parent
                        Path.cwd() / "config",  # Current working directory
                    ]
                    
                    for config_path in possible_config_dirs:
                        if config_path.exists():
                            print(f"Found config directory at: {config_path.resolve()}")
                            # Update working directory to parent of config
                            working_dir = config_path.parent
                            break
                    else:
                        print(f"Warning: config directory not found. Searched in:")
                        for cp in [config_dir] + possible_config_dirs:
                            print(f"  - {cp}")
                
                # Run pipeline with correct working directory
                pipeline_result = subprocess.run(
                    [sys.executable, str(pipeline_path)], 
                    capture_output=True, 
                    text=True, 
                    cwd=str(working_dir)
                )
                pipeline_complete = True
            except Exception as e:
                pipeline_result = e
                pipeline_complete = True
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=run_pipeline)
        pipeline_thread.start()
        
        # Animate progress bar - much slower for 2-3 minute process
        start_time = time.time()
        while not pipeline_complete:
            elapsed_time = time.time() - start_time
            
            # Slower, more realistic progress calculation
            # Assume 150 seconds (2.5 minutes) total time
            expected_duration = 150  # seconds
            time_based_progress = min(int((elapsed_time / expected_duration) * 90), 90)  # Cap at 90%
            
            # Gradual increment with some randomness
            if progress_value < time_based_progress:
                progress_value = min(progress_value + random.uniform(0.5, 1.5), time_based_progress)
            
            # Update progress bar
            progress_bar.progress(int(progress_value))
            
            # Update message based on progress
            message_stage = min(int(progress_value / 4), len(self.progress_messages) - 1)
            if message_stage != message_index:
                message_index = message_stage
            
            current_message = self.progress_messages[message_index]
            
            # Add animated dots and time info
            dots = "." * ((int(elapsed_time * 2) % 3) + 1)
            elapsed_str = f"{int(elapsed_time//60)}:{int(elapsed_time%60):02d}"
            
            status_text.text(f"{current_message}{dots} ({int(progress_value)}%) - {elapsed_str}")
            
            time.sleep(0.8)  # Slower update interval
        
        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text("🎉 Pipeline completed successfully! (100%)")
        time.sleep(1)  # Brief pause to show completion
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Return pipeline result
        pipeline_thread.join()  # Ensure thread completes
        return pipeline_result


class GitManager:
    """Handles Git operations and GitHub integration."""
    
    @staticmethod
    def setup_git_credentials(github_token, username, email):
        """Setup git credentials for authentication"""
        try:
            # Configure git user (use --global to ensure it's set)
            subprocess.run(['git', 'config', '--global', 'user.name', username], check=True)
            subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
            
            # Also set local config as backup
            subprocess.run(['git', 'config', 'user.name', username], check=True)
            subprocess.run(['git', 'config', 'user.email', email], check=True)
            
            # Setup token authentication
            subprocess.run([
                'git', 'config', 'credential.helper', 
                f'store --file=/tmp/git-credentials'
            ], check=True)
            
            # Store credentials
            with open('/tmp/git-credentials', 'w') as f:
                f.write(f'https://{username}:{github_token}@github.com\n')
            
            return True
        except subprocess.CalledProcessError as e:
            return False

    @staticmethod
    def push_to_github_safe(file_paths, commit_message):
        """
        Safely push files to GitHub with proper error handling
        Returns (success: bool, message: str)
        """
        try:
            # Get credentials from Streamlit secrets
            github_token = st.secrets.get("GITHUB_TOKEN", "")
            github_username = st.secrets.get("GITHUB_USERNAME", "")
            github_email = st.secrets.get("GITHUB_EMAIL", "")
            
            if not all([github_token, github_username, github_email]):
                return False, "GitHub credentials not configured in Streamlit secrets"
            
            # Setup credentials
            if not GitManager.setup_git_credentials(github_token, github_username, github_email):
                return False, "Failed to setup git credentials"
            
            # Add files
            files_added = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    result = subprocess.run(['git', 'add', file_path], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        files_added.append(file_path)
                    else:
                        return False, f"Failed to add {file_path}: {result.stderr}"
                # Don't fail if file doesn't exist, just skip it
            
            if not files_added:
                return True, "No files to commit"
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'], 
                                  capture_output=True)
            if result.returncode == 0:
                return True, "No changes to commit"
            
            # Commit changes with explicit author
            result = subprocess.run([
                'git', 'commit', 
                '-m', commit_message,
                '--author', f'{github_username} <{github_email}>'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Failed to commit: {result.stderr}"
            
            # Push to GitHub
            result = subprocess.run([
                'git', 'push', 'origin', 'main'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Failed to push: {result.stderr}"
            
            return True, f"Successfully committed and pushed {len(files_added)} files"
            
        except Exception as e:
            return False, f"Error during git operations: {str(e)}"
        finally:
            # Clean up credentials file
            if os.path.exists('/tmp/git-credentials'):
                try:
                    os.remove('/tmp/git-credentials')
                except:
                    pass

    @staticmethod
    def auto_push_to_github():
        """
        Enhanced version of auto_push_to_github function
        Returns (success: bool, message: str)
        """
        # Define files that might be created/updated by your pipeline
        potential_files = [
            "data/processed_data.csv",
            "data/weather_data.csv", 
            "data/energy_data.csv",
            "data/analysis_results.json",
            "logs/pipeline.log",
            "results/",
        ]
        
        # Create commit message with timestamp
        commit_msg = f"Auto-update data pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return GitManager.push_to_github_safe(potential_files, commit_msg)


class PipelineOrchestrator:
    """Main orchestrator for pipeline operations."""
    
    def __init__(self):
        self.pipeline_runner = PipelineRunner()
        self.git_manager = GitManager()
    
    def handle_pipeline_execution(self, is_manual=False):
        """Handle pipeline execution with proper error handling and Git integration."""
        import time
        
        # Create placeholder for the refresh message
        if is_manual:
            refresh_placeholder = st.empty()
            refresh_placeholder.info("🚀 Manually refreshing data pipeline...")
        else:
            refresh_placeholder = st.empty()
            refresh_placeholder.info("🚀 Auto-updating data pipeline (data is 3+ days old)...")
        
        # Run pipeline with animated progress
        result = self.pipeline_runner.run_pipeline_with_progress()
        
        # Clear the refresh message
        refresh_placeholder.empty()
        
        # Create placeholder for success/error message
        result_placeholder = st.empty()
        
        if isinstance(result, Exception):
            result_placeholder.error(f"❌ Pipeline failed: {str(result)}")
            return False
        elif result.returncode == 0:
            result_placeholder.success("✅ Data pipeline completed successfully!")
            st.balloons()  # Celebration animation!
            
            # Set session state for success message
            st.session_state.show_success_message = True
            st.session_state.message_timestamp = time.time()
            
            # Attempt automatic GitHub push with improved error handling
            push_status_placeholder = st.empty()
            push_status_placeholder.info("📤 Pushing updated data to GitHub...")
            
            git_success, git_message = self.git_manager.auto_push_to_github()
            push_status_placeholder.empty()  # Clear the pushing message
            
            if git_success:
                if "No changes to commit" not in git_message and "No files to commit" not in git_message:
                    st.success("🚀 Successfully pushed updated data to GitHub!")
                    st.info("🔄 Your Streamlit app will redeploy automatically with new data")
                else:
                    st.info("📝 Pipeline completed - no new changes to push to GitHub")
            else:
                st.warning(f"⚠️ Pipeline succeeded but GitHub push failed: {git_message}")
                st.info("💡 You may need to check your GitHub credentials in Streamlit secrets")
                
                # Add debug button for troubleshooting
                if st.button("🔍 Debug GitHub Config", key="debug_git"):
                    st.write("**GitHub Configuration Debug:**")
                    token_present = bool(st.secrets.get("GITHUB_TOKEN", ""))
                    username_present = bool(st.secrets.get("GITHUB_USERNAME", ""))
                    email_present = bool(st.secrets.get("GITHUB_EMAIL", ""))
                    
                    st.write(f"- Token configured: {'✅' if token_present else '❌'}")
                    st.write(f"- Username configured: {'✅' if username_present else '❌'}")
                    st.write(f"- Email configured: {'✅' if email_present else '❌'}")
                    
                    if not all([token_present, username_present, email_present]):
                        st.error("Please configure all GitHub credentials in Streamlit secrets:")
                        st.code("""
GITHUB_TOKEN = "your_github_token_here"
GITHUB_USERNAME = "your_github_username"
GITHUB_EMAIL = "your_email@example.com"
                        """)
            
            return True
        else:
            result_placeholder.error(f"❌ Pipeline failed: {result.stderr}")
            return False
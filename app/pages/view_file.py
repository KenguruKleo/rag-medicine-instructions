"""
Streamlit page for viewing medical instruction files (HTML/MHT).

This page displays the content of HTML and MHT files from the data directory.
"""

import os
import email
import quopri
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import streamlit as st
from dotenv import load_dotenv

# Page configuration MUST be first
st.set_page_config(
    page_title="View Medical Instruction",
    page_icon="📄",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))


def parse_mht_file(mht_path: Path) -> Optional[str]:
    """Parse MHT file and extract HTML content"""
    try:
        with open(mht_path, "rb") as f:
            msg = email.message_from_bytes(f.read())

        # Find the main HTML part
        html_content = None
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/html":
                # Get encoding
                encoding = part.get_content_charset() or "utf-8"
                transfer_encoding = part.get("Content-Transfer-Encoding", "").lower()

                # Get payload
                if transfer_encoding == "quoted-printable":
                    # Handle quoted-printable encoding
                    payload = part.get_payload()
                    if isinstance(payload, str):
                        # Decode quoted-printable
                        payload = quopri.decodestring(payload.encode()).decode(encoding, errors="ignore")
                        html_content = payload
                    else:
                        payload = part.get_payload(decode=True)
                        if payload:
                            try:
                                html_content = payload.decode(encoding)
                            except UnicodeDecodeError:
                                html_content = payload.decode("utf-8", errors="ignore")
                else:
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            html_content = payload.decode(encoding)
                        except UnicodeDecodeError:
                            try:
                                html_content = payload.decode("windows-1251")
                            except UnicodeDecodeError:
                                html_content = payload.decode("utf-8", errors="ignore")
                break

        return html_content
    except Exception as e:
        st.error(f"Error parsing MHT file: {e}")
        return None


def read_html_file(file_path: Path) -> Optional[str]:
    """Read HTML file content."""
    try:
        # Try UTF-8 first
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try windows-1251
            try:
                return file_path.read_text(encoding="windows-1251")
            except Exception:
                # Fallback to UTF-8 with errors replaced
                return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        st.error(f"Error reading HTML file: {e}")
        return None


def main():
    """Main page function."""
    st.title("📄 Medical Instruction Viewer")
    
    # Get file path from query parameters
    file_path_param = st.query_params.get("file", None)
    
    if not file_path_param:
        st.error("No file specified. Please provide a file path in the URL.")
        st.info("Example: ?file=data/html/example.html")
        return
    
    # Decode URL-encoded path
    file_path = unquote(file_path_param)
    
    # Construct full path - handle both relative and absolute paths
    if Path(file_path).is_absolute():
        full_path = Path(file_path)
    else:
        # Relative path - resolve from current working directory
        full_path = (Path.cwd() / file_path).resolve()
    
    # Security check: ensure file is within data directory
    data_dir_abs = (Path.cwd() / DATA_DIR).resolve()
    try:
        full_path.resolve().relative_to(data_dir_abs)
    except ValueError:
        st.error("❌ Access denied: File must be within the data directory.")
        st.info(f"File path: {full_path}")
        st.info(f"Data directory: {data_dir_abs}")
        return
    
    # Check if file exists
    if not full_path.exists():
        st.error(f"❌ File not found: {file_path}")
        st.info(f"Tried path: {full_path}")
        st.info(f"Current working directory: {Path.cwd()}")
        return
    
    # Determine file type
    file_ext = full_path.suffix.lower()
    
    if file_ext == ".mht":
        # Parse MHT file
        html_content = parse_mht_file(full_path)
        if not html_content:
            st.error("Could not extract HTML content from MHT file.")
            return
    elif file_ext in [".html", ".htm"]:
        # Read HTML file
        html_content = read_html_file(full_path)
        if not html_content:
            st.error("Could not read HTML file.")
            return
    else:
        st.error(f"Unsupported file type: {file_ext}")
        return
    
    # Display file info
    st.info(f"📄 **File:** `{full_path.name}` | **Path:** `{file_path}`")
    
    # Display HTML content
    st.components.v1.html(html_content, height=800, scrolling=True)


# Call main function when page loads
main()


"""
Helper functions for file parsing and text extraction.

This module contains utility functions used across the application:
- parse_mht_file: Parse MHT files and extract HTML content
- extract_text_from_html: Extract clean text from HTML content
"""

import email
import email.policy
import logging
import quopri
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_mht_file(mht_path: Path) -> Optional[str]:
    """Parse MHT file and extract HTML content.
    
    Args:
        mht_path: Path to the MHT file
        
    Returns:
        HTML content as string, or None if parsing fails
    """
    try:
        with open(mht_path, "rb") as f:
            msg = email.message_from_bytes(f.read(), policy=email.policy.default)

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
        logger.error(f"Error parsing MHT file {mht_path}: {e}")
        return None


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        Clean text extracted from HTML
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        # Get text
        text = soup.get_text(separator=" ", strip=True)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        return ""


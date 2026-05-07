"""
Contact detail extractor.

Given a page's HTML (or scraped Markdown), extracts:
  - Phone numbers
  - Email addresses
  - Physical address hints
  - WhatsApp numbers

Uses regex patterns robust enough for international formats.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Phone: supports +971 4 888 1050, 04-8886571, (04) 888-6571, +1-800-123-4567, etc.
_PHONE_RE = re.compile(
    r"""
    (?:
        \+?[\d\s\-().]{7,20}   # international / local formats
    )
    """,
    re.VERBOSE,
)

# Stricter phone validator — must have at least 7 consecutive digits
_PHONE_DIGITS_RE = re.compile(r'\+?[\d\s\-().]{7,20}')
_MIN_DIGITS = 7

# Email
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# WhatsApp — look for "whatsapp" near a number on the same line
_WHATSAPP_RE = re.compile(
    r"(?i)whatsapp[:\s]*([+\d\s\-().]{7,20})"
)

# Address hints — lines containing street-like keywords
_ADDRESS_KEYWORDS = re.compile(
    r"(?i)\b(street|st\.|road|rd\.|avenue|ave\.|blvd|boulevard|floor|"
    r"office|building|tower|suite|district|zone|area|block|plot|"
    r"p\.?o\.?\s*box|po box|freezone|free zone|industrial|business "
    r"bay|downtown|city|dubai|abu dhabi|sharjah|ajman|mumbai|delhi|"
    r"bangalore|chennai|kolkata|hyderabad)\b"
)


def _clean_phone(raw: str) -> Optional[str]:
    """Return *raw* if it looks like a real phone number, else None."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) < _MIN_DIGITS:
        return None
    # Avoid matching things like years (1990, 2024) or short codes
    if len(digits) > 15:
        return None
    return raw.strip()


def extract_contacts_from_html(html: str, base_url: str = "") -> dict:
    """Extract contact details from raw HTML.

    Returns
    -------
    dict with keys:
        phones   : list[str]
        emails   : list[str]
        addresses: list[str]
        whatsapp : list[str]
    """
    soup = BeautifulSoup(html, "html.parser")

    # Grab all visible text
    text = soup.get_text(separator="\n")

    phones: list[str]    = []
    emails: list[str]    = []
    addresses: list[str] = []
    whatsapp: list[str]  = []

    # ── Emails ────────────────────────────────────────────────────────
    # Also check mailto: links — most reliable source
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            em = href[7:].split("?")[0].strip()
            if em and em not in emails:
                emails.append(em)
        if href.startswith("tel:"):
            ph = href[4:].strip()
            cleaned = _clean_phone(ph)
            if cleaned and cleaned not in phones:
                phones.append(cleaned)

    for m in _EMAIL_RE.finditer(text):
        em = m.group(0).strip()
        if em not in emails:
            emails.append(em)

    # ── WhatsApp ──────────────────────────────────────────────────────
    for m in _WHATSAPP_RE.finditer(text):
        wa = m.group(1).strip()
        cleaned = _clean_phone(wa)
        if cleaned and cleaned not in whatsapp:
            whatsapp.append(cleaned)

    # ── Phones ────────────────────────────────────────────────────────
    # Look for tel: links first (most reliable)
    # Then scan lines that contain phone-like patterns near keywords
    phone_context_re = re.compile(
        r"(?i)(?:phone|tel|telephone|call|mob|mobile|contact|fax|hotline)"
        r"[^\n]{0,50}"
        r"(\+?[\d][\d\s\-().]{5,18}[\d])"
    )
    for m in phone_context_re.finditer(text):
        raw = m.group(1)
        cleaned = _clean_phone(raw)
        if cleaned and cleaned not in phones and cleaned not in whatsapp:
            phones.append(cleaned)

    # Broader phone scan as fallback
    for line in text.splitlines():
        # Only process lines that look contact-related
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["phone", "tel", "call", "mob", "fax", "+", "04-", "04 "]):
            for m in _PHONE_DIGITS_RE.finditer(line):
                raw = m.group(0)
                cleaned = _clean_phone(raw)
                if cleaned and cleaned not in phones and cleaned not in whatsapp:
                    phones.append(cleaned)

    # ── Address ───────────────────────────────────────────────────────
    for line in text.splitlines():
        if _ADDRESS_KEYWORDS.search(line):
            cleaned_line = line.strip()
            if 10 < len(cleaned_line) < 200 and cleaned_line not in addresses:
                addresses.append(cleaned_line)

    return {
        "phones":    phones[:5],
        "emails":    emails[:5],
        "addresses": addresses[:3],
        "whatsapp":  whatsapp[:3],
    }


def extract_contacts_from_url(url: str, timeout: int = 10) -> dict:
    """Fetch *url* and extract contact details.

    Returns the same dict as :func:`extract_contacts_from_html`,
    plus ``"success": bool``.
    """
    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
        contacts = extract_contacts_from_html(resp.text, base_url=url)
        contacts["success"] = True
        logger.info(
            "Contact extraction from %s: %d phones, %d emails, %d addresses",
            url, len(contacts["phones"]), len(contacts["emails"]), len(contacts["addresses"]),
        )
        return contacts
    except Exception as exc:
        logger.warning("Contact extraction failed for %s: %s", url, exc)
        return {"phones": [], "emails": [], "addresses": [], "whatsapp": [], "success": False}

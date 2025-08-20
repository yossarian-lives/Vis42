"""
JSON Utilities

Helper functions for parsing and handling JSON responses from LLM providers.
"""

import json
import re
from typing import Optional, Dict, Any

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse strict JSON. If it fails, attempt to repair by extracting the
    first balanced-looking {...} block.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None 
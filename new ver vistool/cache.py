"""
Simple caching layer for LLM visibility results
"""
from __future__ import annotations

import time
import json
from typing import Any, Dict, Optional

# In-memory cache with TTL
_cache: Dict[str, Dict[str, Any]] = {}


async def get_cache(key: str) -> Optional[Dict[str, Any]]:
    """Get cached value if it exists and hasn't expired"""
    if key not in _cache:
        return None
    
    entry = _cache[key]
    if time.time() > entry["expires_at"]:
        # Expired, remove it
        del _cache[key]
        return None
    
    return entry["data"]


async def set_cache(key: str, data: Dict[str, Any], ttl: int) -> None:
    """Set cached value with TTL in seconds"""
    _cache[key] = {
        "data": data,
        "expires_at": time.time() + ttl,
        "created_at": time.time()
    }


async def clear_cache() -> None:
    """Clear all cached data"""
    _cache.clear()


async def cleanup_expired() -> int:
    """Remove expired entries from cache, return number removed"""
    now = time.time()
    expired_keys = [
        key for key, entry in _cache.items()
        if now > entry["expires_at"]
    ]
    
    for key in expired_keys:
        del _cache[key]
    
    return len(expired_keys)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    now = time.time()
    active_entries = sum(1 for entry in _cache.values() if now <= entry["expires_at"])
    expired_entries = len(_cache) - active_entries
    
    return {
        "total_entries": len(_cache),
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "cache_size_bytes": sum(
            len(json.dumps(entry["data"], default=str)) 
            for entry in _cache.values()
        )
    }
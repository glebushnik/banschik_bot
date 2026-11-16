def split_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    """Overlapping Unicode-safe chunks, preferring whitespace boundaries."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Expected size > overlap >= 0")
    text = text.strip()
    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result

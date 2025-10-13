import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.

    Args:
        filename (str): Original filename

    Returns:
        str: Sanitized filename
    """
    # Replace problematic characters
    filename = filename.replace(":", " -")
    filename = filename.replace("/", "-")
    filename = filename.replace("\\", "-")

    # Remove any other invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)

    return filename.strip()

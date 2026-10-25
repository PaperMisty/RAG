import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def summarize_docs(docs, limit=3):
    """Log-friendly summary of retrieved/reranked docs.

    Only titles, file names and scores are logged, never full chunk
    content: the old json_format dumps made the log unreadable and
    grew it by tens of kilobytes per query.
    """
    docs = docs or []
    items = []
    for doc in docs[:limit]:
        if not isinstance(doc, dict):
            items.append(str(doc)[:60])
            continue
        title = doc.get("title") or doc.get("section_title") or doc.get("file_title") or "?"
        items.append(f"{title} score={doc.get('score')}")
    return f"count={len(docs)} top=[{'; '.join(items)}]"

if not logger.handlers:
    if HAS_COLORLOG:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(handler)
    # Persistent file log: project-root logs/ (not /tmp, not the terminal).
    # Plain text, no ANSI colors, rotated so it cannot grow unbounded.
    # If the directory cannot be created, the service must still start
    # on console logging alone.
    try:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "kb_frontend.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
    except Exception:
        pass

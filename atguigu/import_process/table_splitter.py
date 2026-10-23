"""表格 chunk 按行拆分（用户 2026-10-23 规格）。

背景：NodeDocumentSplit.split_chunks 对含表格的 section 整段成一个
chunk、绕过 300 字上限。长表 chunk 有两层截断风险：
  1) Milvus chunk_content 的 varchar(65535) 超限时整批 insert 硬报错
     （MilvusException code=1100，2026-10-23 天合光能小样实测），
     一份文档整份入库失败；
  2) BGE-M3 超过 8192 tokens 时在编码侧静默截断，向量只覆盖前段，
     表尾内容对检索不可见且无任何告警。
本模块把超长表格按“完整行”拆成 <=300 字的结构完整迷你表：
HTML 表按 <tr>...</tr> 边界切，片段补 </table> 闭合、后续片段
重复表头 <tr>；Markdown 管道表按整行切，后续片段重复表头行与
分隔行。非表格文本不处理；没有任何一张表超限时原样返回，
普通切分路径行为不变。

TODO(长表回归): 当前语料里没有真正触发极端长表的文档，本机制
先行落地；真长表文档出现时需用 table_needle_test 复测完整召回。
"""

import re

from atguigu.tool.logger import logger

DEFAULT_MAX_LEN = 300

_TABLE_SPAN_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TH_RE = re.compile(r"<th\b", re.IGNORECASE)
_MD_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*$")
_TAG_RE = re.compile(r"<[^>]+>")


def _is_pipe_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _lines_to_blocks(text: str) -> list[tuple[str, object]]:
    """把一段纯文本切成 ("text", str) 与 ("md", list[str]) 块。

    连续的管道行构成候选 MD 表；候选块第二行不是分隔行时整体
    回退为文本行，避免误伤含竖线的普通段落。
    """
    blocks: list[tuple[str, object]] = []
    text_lines: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if _is_pipe_line(lines[i]):
            j = i
            while j < len(lines) and _is_pipe_line(lines[j]):
                j += 1
            run = lines[i:j]
            if len(run) >= 2 and _MD_SEP_RE.match(run[1]):
                if text_lines:
                    blocks.append(("text", "\n".join(text_lines)))
                    text_lines = []
                blocks.append(("md", run))
            else:
                text_lines.extend(run)
            i = j
        else:
            text_lines.append(lines[i])
            i += 1
    if text_lines:
        blocks.append(("text", "\n".join(text_lines)))
    return blocks


def _split_segments(content: str) -> list[tuple[str, object]]:
    """先按 HTML 表格跨度切开正文，间隙文本再细分为 text/md 块。"""
    blocks: list[tuple[str, object]] = []
    pos = 0
    for m in _TABLE_SPAN_RE.finditer(content):
        if m.start() > pos:
            blocks.extend(_lines_to_blocks(content[pos : m.start()]))
        blocks.append(("html", m.group(0)))
        pos = m.end()
    if pos < len(content):
        blocks.extend(_lines_to_blocks(content[pos:]))
    return blocks


def _split_html_table(table: str, max_len: int, first_budget: int | None = None) -> list[str]:
    """把一张 HTML 表按完整 <tr> 行拆成结构完整的迷你表片段。

    first_budget：第一片的字符预算（给待拼接的前置文本留位），
    None 表示与 max_len 相同。单行本身放不下时整行成片（不破行），
    该片可能超限——这是不破完整行约束下的最优结果。
    """
    rows = _TR_RE.findall(table)
    if not rows:
        return [table]
    open_m = _TABLE_OPEN_RE.search(table)
    open_tag = open_m.group(0) if open_m else "<table>"
    # 行外文本（如 caption 里的文字）不丢弃，单独作为文本片返回在最前
    outside = _TABLE_CLOSE_RE.sub("", _TABLE_OPEN_RE.sub("", _TR_RE.sub("", table)))
    outside_text = _TAG_RE.sub("", outside).strip()

    n_header = 0
    while n_header < len(rows) and _TH_RE.search(rows[n_header]):
        n_header += 1
    if n_header == 0:
        header_rows, data_rows = rows[:1], rows[1:]
    elif n_header >= len(rows):
        header_rows, data_rows = rows[:1], rows[1:]
    else:
        header_rows, data_rows = rows[:n_header], rows[n_header:]

    header = "".join(header_rows)
    prefix = open_tag + header
    suffix = "</table>"
    if not data_rows:
        return [table]

    pieces: list[str] = []
    cur = prefix
    budget = first_budget if first_budget is not None else max_len
    for row in data_rows:
        if len(cur) + len(row) + len(suffix) > budget and cur != prefix:
            pieces.append(cur + suffix)
            cur = prefix
            budget = max_len
        cur += row
    pieces.append(cur + suffix)
    if outside_text:
        pieces.insert(0, outside_text)
    return pieces


def _split_md_table(lines: list[str], max_len: int, first_budget: int | None = None) -> list[str]:
    """把 MD 管道表按整行拆分，后续片段重复表头行+分隔行。"""
    header, sep = lines[0], lines[1]
    data_rows = lines[2:]
    if not data_rows:
        return ["\n".join(lines)]
    prefix = header + "\n" + sep
    pieces: list[str] = []
    cur_lines = [header, sep]
    cur_len = len(prefix)
    budget = first_budget if first_budget is not None else max_len
    for row in data_rows:
        add = len(row) + 1
        if cur_len + add > budget and len(cur_lines) > 2:
            pieces.append("\n".join(cur_lines))
            cur_lines = [header, sep]
            cur_len = len(prefix)
            budget = max_len
        cur_lines.append(row)
        cur_len += add
    pieces.append("\n".join(cur_lines))
    return pieces


def split_table_content(content: str, max_len: int = DEFAULT_MAX_LEN) -> list[str]:
    """把含超长表格的 chunk 正文按完整行拆为 <=max_len 的片段列表。

    触发条件：正文总长超过 max_len，且其中至少一张表（HTML 或 MD）
    本身超过 max_len。未触发时返回 [content] 原样（普通文档与
    小表行为与拆分前完全一致）。表格前后的普通文本优先拼到相邻
    表片段上（放不下时自成一片），不丢内容。
    """
    if len(content) <= max_len:
        return [content]
    blocks = _split_segments(content)
    table_blocks = [(k, b) for k, b in blocks if k in ("html", "md")]
    if not table_blocks:
        return [content]

    def block_len(block) -> int:
        return len(block) if isinstance(block, str) else len("\n".join(block))

    if not any(block_len(b) > max_len for _, b in table_blocks):
        return [content]

    fragments: list[str] = []
    pending_text: str | None = None

    def flush_pending_standalone() -> None:
        nonlocal pending_text
        if pending_text and pending_text.strip():
            fragments.append(pending_text)
        pending_text = None

    for kind, block in blocks:
        if kind == "text":
            text = block  # type: ignore[assignment]
            if text.strip():
                pending_text = text if pending_text is None else pending_text + "\n" + text
            continue
        # 表格块：先算行拆分（第一片给 pending 文本预留预算）
        reserve = len(pending_text) + 1 if pending_text else 0
        first_budget = max_len - reserve if reserve else None
        if block_len(block) > max_len:
            if kind == "html":
                pieces = _split_html_table(block, max_len, first_budget)  # type: ignore[arg-type]
            else:
                pieces = _split_md_table(block, max_len, first_budget)  # type: ignore[arg-type]
        else:
            pieces = ["\n".join(block) if kind == "md" else block]  # type: ignore[arg-type]
        # 行外文本片（caption 等）排在表片之前，先并入 pending
        while pieces and not pieces[0].lstrip().startswith(("<table", "|")):
            head = pieces.pop(0)
            pending_text = head if pending_text is None else pending_text + "\n" + head
        if not pieces:
            continue
        if pending_text is not None:
            if len(pending_text) + 1 + len(pieces[0]) <= max_len:
                pieces[0] = pending_text + "\n" + pieces[0]
                pending_text = None
            else:
                flush_pending_standalone()
        fragments.extend(pieces)

    if pending_text and pending_text.strip():
        if fragments and len(fragments[-1]) + 1 + len(pending_text) <= max_len:
            fragments[-1] = fragments[-1] + "\n" + pending_text
        else:
            fragments.append(pending_text)

    if len(fragments) == 1 and fragments[0] == content:
        return [content]
    logger.info(
        f"表格行拆分：{len(content)} 字 -> {len(fragments)} 片（上限 {max_len} 字/片）"
    )
    return fragments


def expand_chunk_dict(chunk: dict, fragments: list[str]) -> list[dict]:
    """把一个 chunk 字典按正文片段展开为多个 chunk 字典。

    继承 file_title/section_title/page/item_name 等全部元数据；
    未真正拆分（单片段且与原文相同）时原样返回，向量字段保留。
    真拆分时：part 按既有约定顺延编号 1..N（未切分为 0、切分
    片段为 1 起，与 RecursiveCharacterTextSplitter 路径一致）；
    dense/sparse/id 失效，一律剔除，由调用方重新向量化。
    """
    if len(fragments) == 1 and fragments[0] == chunk.get("chunk_content"):
        return [chunk]
    out: list[dict] = []
    for idx, frag in enumerate(fragments, 1):
        new_chunk = dict(chunk)
        new_chunk["chunk_content"] = frag
        new_chunk["part"] = idx
        for stale in ("dense", "sparse", "id"):
            new_chunk.pop(stale, None)
        out.append(new_chunk)
    return out

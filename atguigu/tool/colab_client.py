"""Colab 推理栈网关客户端（CFQA/Colab 接入）。

Colab L4 上部署了与本地同构的推理栈，经聚合 FastAPI + Cloudflare 隧道对外：
- MinerU 解析服务（网关 POST /parse 转发）
- GLM-OCR（网关 /ocr，本模块暂不涉及）
- BGE-M3 向量化服务（FlagEmbedding 实现，TEI 契约：POST /embed 出 dense、
  POST /embed_sparse 出 sparse）

本模块提供两个与既有工具对齐的能力：
1. colab_embed(texts)：签名与返回结构与 bgem3_client_tool / bgem3_http_tool 的
   get_bgem3_embedding 完全一致（{"dense": [[...]], "sparse": [{int: float}]}），
   由 bgem3_tool 按 EMBED_SOURCE=colab 时分发到这里；
2. colab_parse_pdf(pdf_path)：上传 PDF 到网关 /parse，拿回 markdown；
   若响应带 MinerU content_list（含 page_idx），同时生成带页码标记的
   markdown（每页首块前插入 <!-- page:N -->，N 为 1 起的 PDF 页序），
   供切分节点给 chunk 标注页码——这是 CFQA 金标页级评测的前提。

网关地址只来自 .env 的 COLAB_BASE_URL（隧道重启地址会变，不写死在代码里）。
"""

import re
import time
from pathlib import Path

import requests

from atguigu.config.config import ColabConfig
from atguigu.tool.logger import logger

# 与切分节点 node_document_split 约定的页码标记格式（HTML 注释，渲染不可见）
PAGE_MARKER_TEMPLATE = "<!-- page:{page} -->"
PAGE_MARKER_RE = re.compile(r"^<!--\s*page:(\d+)\s*-->$")

_session = requests.Session()
# 联调实测（2026-10-23）：经用户侧 Clash fake-ip + Cloudflare 隧道的长连接
# 池会半死（请求已发出、响应永不到达，进程整段阻塞在 ssl read，urllib3 的
# 读超时按单次 recv 计也救不了）。对策：禁用连接复用，每次请求新建连接，
# 开销（握手 ~0.3s）远小于推理耗时，可接受。
_session.headers.update({"Connection": "close"})

# TEI 单请求批量上限各部署不一，客户端先按 32 分批，稳妥优先
_EMBED_BATCH = 32

# 网关经 Cloudflare 隧道偶发 502（2026-10-23 联调实测：并发/负载下出现），
# 对幂等 POST 做有限重试，避免整批导入被单次抖动打断
_RETRY_STATUSES = (502, 503, 504)
_RETRY_BACKOFFS = (5, 15, 30)


def _post_with_retry(url: str, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(len(_RETRY_BACKOFFS) + 1):
        try:
            resp = _session.post(url, **kwargs)
            if resp.status_code not in _RETRY_STATUSES or attempt == len(_RETRY_BACKOFFS):
                return resp
            logger.warning(
                f"POST {url} 返回 {resp.status_code}，第 {attempt + 1} 次重试前等待 {_RETRY_BACKOFFS[attempt]}s"
            )
        except requests.RequestException as e:
            last_exc = e
            if attempt == len(_RETRY_BACKOFFS):
                raise
            logger.warning(f"POST {url} 异常 {e}，第 {attempt + 1} 次重试前等待 {_RETRY_BACKOFFS[attempt]}s")
        time.sleep(_RETRY_BACKOFFS[attempt])
    raise last_exc  # pragma: no cover


def _base_url() -> str:
    base = (ColabConfig.colab_base_url or "").rstrip("/")
    if not base:
        raise RuntimeError(
            "COLAB_BASE_URL 未配置：请在 .env 中填入 Colab 网关的 Cloudflare 隧道地址"
        )
    return base


def colab_health() -> dict:
    """网关健康检查：GET /health，返回网关聚合的各服务状态。"""
    resp = _session.get(
        f"{_base_url()}/health", timeout=(10, ColabConfig.colab_timeout)
    )
    resp.raise_for_status()
    return resp.json()


def _unwrap_list(data, *keys):
    """TEI 原生返回顶层 list；聚合网关可能包一层 dict，按候选键依次拆解。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"无法识别的响应结构: {type(data)} / {list(data) if isinstance(data, dict) else ''}")


def _embed_batch(texts: list[str]) -> dict:
    """单批向量化：/embed 拿 dense，/embed_sparse 拿 sparse，拼成统一结构。"""
    from atguigu.tool.bgem3_http_tool import _sparse_item_to_dict

    base = _base_url()
    payload = {"inputs": texts, "truncate": True}
    resp = _post_with_retry(
        f"{base}/embed", json=payload, timeout=(10, ColabConfig.colab_timeout)
    )
    resp.raise_for_status()
    dense_raw = _unwrap_list(resp.json(), "embeddings", "data", "dense")
    dense = [[float(x) for x in vec] for vec in dense_raw]

    resp = _post_with_retry(
        f"{base}/embed_sparse", json=payload, timeout=(10, ColabConfig.colab_timeout)
    )
    resp.raise_for_status()
    sparse_raw = _unwrap_list(resp.json(), "embeddings", "data", "sparse")
    sparse = [_sparse_item_to_dict(item) for item in sparse_raw]

    if len(dense) != len(texts) or len(sparse) != len(texts):
        raise ValueError(
            f"网关返回条数与输入不一致: inputs={len(texts)} dense={len(dense)} sparse={len(sparse)}"
        )
    return {"dense": dense, "sparse": sparse}


def colab_embed(texts: list[str]) -> dict:
    """把文本列表转为稀疏和稠密向量（Colab 版，签名与本地/远程版一致）。"""
    dense_all: list = []
    sparse_all: list = []
    for i in range(0, len(texts), _EMBED_BATCH):
        part = _embed_batch(texts[i : i + _EMBED_BATCH])
        dense_all.extend(part["dense"])
        sparse_all.extend(part["sparse"])
    return {"dense": dense_all, "sparse": sparse_all}


def _find_key(obj, keys: tuple[str, ...], want_type):
    """在嵌套 dict/list 中深度优先找第一个类型匹配的候选键（MinerU 各版本响应结构不一）。"""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and isinstance(obj[key], want_type):
                return obj[key]
        for value in obj.values():
            found = _find_key(value, keys, want_type)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key(item, keys, want_type)
            if found is not None:
                return found
    return None


def _block_text(block: dict) -> str:
    """从 MinerU content_list 的一个块中抽取可入 md 的文本。"""
    btype = block.get("type", "text")
    if btype == "table":
        return block.get("table_body") or block.get("text") or ""
    if btype == "image":
        img_path = block.get("img_path") or ""
        return f"![image]({img_path})" if img_path else ""
    return block.get("text") or ""


def build_page_annotated_md(content_list: list[dict]) -> str:
    """把 MinerU content_list 拼成带页码标记的 markdown。

    page_idx 为 0 起，标记页码 +1 转成 1 起的 PDF 页序，与 CFQA 金标
    "答案出自"（PDF 页序）对齐。空块跳过，标记只在页切换时插入。
    """
    lines: list[str] = []
    current_page = None
    for block in content_list:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx", block.get("page_no", block.get("page", 0)))
        try:
            page = int(page_idx) + 1
        except (TypeError, ValueError):
            page = current_page or 1
        text = _block_text(block).strip()
        if not text:
            continue
        if page != current_page:
            lines.append(PAGE_MARKER_TEMPLATE.format(page=page))
            current_page = page
        lines.append(text)
    return "\n\n".join(lines)


def colab_parse_pdf(pdf_path: str | Path) -> dict:
    """上传 PDF 到网关 /parse 解析。

    Returns:
        dict: {
            "md_content": 网关直接给出的 markdown（可能为空字符串），
            "md_annotated": 由 content_list 拼出的带页码标记 markdown（无 content_list 时等于 md_content），
            "content_list": MinerU content_list（可能为 None），
            "pages": content_list 覆盖的最大页序（1 起；未知为 0），
        }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    base = _base_url()
    pdf_bytes = pdf_path.read_bytes()

    resp = None
    # 网关文件字段名以联调为准：先试单数 file（FastAPI UploadFile 惯例），
    # 422 时回退 MinerU 原生 API 的复数 files，只重试一次。
    for field_name in ("file", "files"):
        resp = _post_with_retry(
            f"{base}/parse",
            files={field_name: (pdf_path.name, pdf_bytes, "application/pdf")},
            timeout=(10, ColabConfig.colab_parse_timeout),
        )
        if resp.status_code != 422:
            break
        logger.warning(f"/parse 以字段 {field_name} 返回 422，尝试下一种字段名")
    resp.raise_for_status()
    data = resp.json()

    # 网关（MinerU 4.x kit 形态）实测结构：files[0].content.markdown 为正文，
    # 且服务端只输出 markdown（structured_content/middle_json 恒为 null、
    # 请求参数无法开启）——页级结构拿不到，页码对齐由导入驱动用 PDF 文本层
    # 锚点回填（见 cfqa_import.align_chunk_pages）。优先按已知路径精确取，
    # 找不到再走通用深搜兜底。
    md_content = ""
    try:
        md_content = data["files"][0]["content"]["markdown"] or ""
    except (KeyError, IndexError, TypeError):
        md_content = _find_key(data, ("md_content", "markdown", "md", "full_md"), str) or ""
    content_list = _find_key(data, ("content_list", "contentList"), list)

    md_annotated = md_content
    pages = 0
    if content_list:
        md_annotated = build_page_annotated_md(content_list)
        for block in content_list:
            if isinstance(block, dict):
                try:
                    pages = max(pages, int(block.get("page_idx", 0)) + 1)
                except (TypeError, ValueError):
                    pass
    logger.info(
        f"colab_parse_pdf {pdf_path.name}: md {len(md_content)} chars, "
        f"annotated {len(md_annotated)} chars, pages {pages}"
    )
    return {
        "md_content": md_content,
        "md_annotated": md_annotated,
        "content_list": content_list,
        "pages": pages,
    }


if __name__ == "__main__":
    print(colab_health())
    res = colab_embed(["hello world", "hello milvus"])
    print("dense dims:", [len(v) for v in res["dense"]])
    print("sparse sizes:", [len(s) for s in res["sparse"]])

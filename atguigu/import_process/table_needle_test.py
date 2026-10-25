"""长表埋针端到端验收（用户 2026-10-23 规格）。

构造一份含 60 行 HTML 长表（3000+ 字）的测试文档：第一条数据行埋
"头针"、最后一条数据行埋"尾针"（虚构科目名 + 金额，全文唯一）。
走真实流水线入库（切分 -> 人工置 item_name -> 本地 BGE 向量化 ->
Milvus 入库，专用集合 kb_table_needle），再用基线 hybrid 检索
（0.8/0.2，与 cfqa_validate 同口径）分别查头针/尾针，报命中名次。

验收标准：尾针能被 top-10 召回。未召回时脚本同时给出全量扫描
结果（针是否在库里），用于定位是拆分/入库问题还是检索问题。

用法：EMBED_SOURCE=local .venv/bin/python -m atguigu.import_process.table_needle_test
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("EMBED_SOURCE", "local")

from atguigu.config.config import MilvusConfig  # noqa: E402
from atguigu.import_process.nodes.node_bge_embedding import NodeBGEEmbedding  # noqa: E402
from atguigu.import_process.nodes.node_document_split import NodeDocumentSplit  # noqa: E402
from atguigu.import_process.nodes.node_import_milvus import NodeImportMilvus  # noqa: E402
from atguigu.tool.bgem3_tool import get_bgem3_embedding  # noqa: E402
from atguigu.tool.milvus_client_tool import (  # noqa: E402
    create_reqs,
    get_milvus_client,
    weighted_hybrid_search,
)

FILE_TITLE = "长表针测文档"
COLLECTION = "kb_table_needle"
TOP_K = 10

HEAD_NEEDLE_NAME = "青藏高原观测站仪器校准费"
HEAD_NEEDLE_AMT = "7310246.18"
TAIL_NEEDLE_NAME = "月球背面中继站维护费"
TAIL_NEEDLE_AMT = "9527481.06"


def build_markdown() -> str:
    header = "<tr><th>科目编号</th><th>科目名称</th><th>期末余额（元）</th></tr>"
    rows = []
    for i in range(1, 61):
        if i == 1:
            name, amt = HEAD_NEEDLE_NAME, HEAD_NEEDLE_AMT
        elif i == 60:
            name, amt = TAIL_NEEDLE_NAME, TAIL_NEEDLE_AMT
        else:
            name, amt = f"第{i:03d}项长期资产明细科目", f"{1_000_000 + i * 137}.00"
        rows.append(f"<tr><td>KM{i:03d}</td><td>{name}</td><td>{amt}</td></tr>")
    table = "<table>" + header + "".join(rows) + "</table>"
    return (
        f"# {FILE_TITLE}\n\n"
        "## 概览\n"
        "本文件为固定资产明细的针测样本，用于验证长表尾部内容能否被完整入库与召回。"
        "表中每一行对应一个科目的期末余额，行数较多，请结合附注阅读。\n\n"
        "## 固定资产明细表\n"
        f"{table}\n\n"
        "## 附注\n"
        "以上余额均为人民币口径，已经年度盘点核对；个别科目为跨期项目，详见附注说明。\n"
    )


def search_needle(query: str, needle_name: str, needle_amt: str) -> dict:
    vecs = get_bgem3_embedding([query])
    reqs = create_reqs(
        dense_data=vecs["dense"][0],
        sparse_data=vecs["sparse"][0],
        dense_ann_field="dense",
        sparse_ann_field="sparse",
        limit=TOP_K,
        expr=f"file_title == '{FILE_TITLE}'",
    )
    res = weighted_hybrid_search(
        collection_name=COLLECTION,
        reqs=reqs,
        ranker=[0.8, 0.2],
        limit=TOP_K,
        output_fields=["page", "file_title", "section_title", "chunk_content", "part"],
    )
    hits = res[0]
    rank = None
    for idx, hit in enumerate(hits, start=1):
        content = hit["entity"].get("chunk_content", "")
        if needle_name in content or needle_amt in content:
            rank = idx
            break
    return {
        "query": query,
        "needle_rank_in_topk": rank,
        "topk_parts": [hit["entity"].get("part") for hit in hits],
    }


def main() -> int:
    md_text = build_markdown()
    out_dir = Path("atguigu/data/output/table_needle")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{FILE_TITLE}.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[needle] 测试文档已生成: {md_path}（{len(md_text)} 字）", flush=True)

    # 1) 真实切分
    chunks = NodeDocumentSplit().process(
        {"md_path": md_path, "file_title": FILE_TITLE}
    )["chunk_dict_list"]
    table_chunks = [c for c in chunks if "固定资产明细表" in (c["section_title"] or "")]
    print(
        f"[needle] 切分完成: 共 {len(chunks)} 个 chunk，表格 section 产出 "
        f"{len(table_chunks)} 个，最大 chunk {max(len(c['chunk_content']) for c in chunks)} 字",
        flush=True,
    )

    # 2) item_name 人工置值（与 cfqa_import 驱动同口径，替代主体识别节点）
    for c in chunks:
        c["item_name"] = "针测主体固定资产"

    # 3) 本地 BGE 向量化（EMBED_SOURCE=local）
    chunks = NodeBGEEmbedding().process(
        {"chunk_dict_list": chunks, "md_path": md_path}
    )["chunk_dict_list"]
    print(f"[needle] 向量化完成: {len(chunks)} 个 chunk", flush=True)

    # 4) 入库（专用测试集合）
    MilvusConfig.chunks_collection = COLLECTION
    NodeImportMilvus().process({"chunk_dict_list": chunks})
    print(f"[needle] 入库完成: 集合 {COLLECTION}", flush=True)

    # 5) 全量扫描核对：针是否真的在库里（与检索命中分开定位）。
    # 注意：新建集合首次 insert 后 query 可见性有短暂延迟（2026-10-23
    # 实测紧贴 insert 的扫描返回 0 行、隔数秒复查即为全量），故带重试。
    client = get_milvus_client()
    stored = []
    for _ in range(6):
        stored = client.query(
            COLLECTION,
            filter=f"file_title == '{FILE_TITLE}'",
            output_fields=["chunk_content"],
            limit=1000,
        )
        if len(stored) >= len(chunks):
            break
        time.sleep(2)
    stored_contents = [r["chunk_content"] for r in stored]
    scan = {
        "stored_chunks": len(stored_contents),
        "head_needle_in_store": any(HEAD_NEEDLE_NAME in c for c in stored_contents),
        "tail_needle_in_store": any(TAIL_NEEDLE_NAME in c for c in stored_contents),
    }
    print(f"[needle] 入库扫描: {json.dumps(scan, ensure_ascii=False)}", flush=True)

    # 6) 基线 hybrid 检索（0.8/0.2）查头针/尾针
    head = search_needle(HEAD_NEEDLE_NAME, HEAD_NEEDLE_NAME, HEAD_NEEDLE_AMT)
    tail = search_needle(TAIL_NEEDLE_NAME, TAIL_NEEDLE_NAME, TAIL_NEEDLE_AMT)
    summary = {
        "collection": COLLECTION,
        "total_chunks": len(chunks),
        "table_chunks": len(table_chunks),
        "max_chunk_chars": max(len(c["chunk_content"]) for c in chunks),
        "scan": scan,
        "head": head,
        "tail": tail,
        "verdict": "PASS" if tail["needle_rank_in_topk"] else "FAIL",
    }
    print("NEEDLE_RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print("NEEDLE_RESULT_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

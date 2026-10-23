"""CFQA 年报集导入驱动（CFQA/Colab 接入）。

全流程：解析（Colab 网关 MinerU）-> 切分（带页码）-> 向量化（Colab 网关 BGE-M3）
-> 入 Milvus（独立集合 cfqa_colab，不污染现有 kb_chunks）。

不走产品手册的图流程：跳过 NodeMDImg（MinIO 图片/VLM 描述）与
NodeItemNameRecognition（LLM 识别主体）——年报场景 item_name 直接用公司名；
其余复用既有节点类 NodeDocumentSplit / NodeBGEEmbedding / NodeImportMilvus。
后端切换通过运行时设置 config 属性完成（不动 .env、不影响主流程默认值），
COLAB_BASE_URL 仍从 .env 读（隧道地址易变）。

解析按页段提交：整本数百页单请求经 Cloudflare 隧道易超时，按
--pages-per-request（默认 40 页）切子 PDF 逐段调网关 /parse，
content_list 的 page_idx 加段偏移后合并，再生成带页码标记的 md。
若某段响应无 content_list，退化为该段 md 前注入段首页标记（段内页码
取段首页，精度下降，汇总 parse_mode 会标注 md_fallback）。

注意：Colab 侧 MinerU 当前为 basic 档（pipeline 小模型），解析质量按
basic 档预期；standard（VLM）档需单独 venv 钉版，待拍板。

用法（项目根目录）：
    .venv/bin/python -m atguigu.import_process.cfqa_import --only cfqa_688599_2022
    .venv/bin/python -m atguigu.import_process.cfqa_import --all --skip-done
产物：OUTPUT_DIR/cfqa/<stem>/ 下的 md、chunks json、embedding json 与
import_summary.json（页数/chunk 数/集合/耗时/解析模式）。
"""

import argparse
import csv
import json
import time
from pathlib import Path

import pymupdf

from atguigu.config.config import (
    OUTPUT_DIR,
    RAW_DIR,
    ColabConfig,
    EmbeddingConfig,
    MilvusConfig,
    MineruConfig,
)
from atguigu.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from atguigu.import_process.nodes.node_document_split import NodeDocumentSplit
from atguigu.import_process.nodes.node_import_milvus import NodeImportMilvus
from atguigu.tool.colab_client import (
    PAGE_MARKER_TEMPLATE,
    build_page_annotated_md,
    colab_health,
    colab_parse_pdf,
)
from atguigu.tool.logger import logger

CFQA_RAW_DIR = RAW_DIR / "cfqa"
CFQA_OUT_DIR = OUTPUT_DIR / "cfqa"


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("status") == "ok" and r.get("filename")]


def parse_pdf_in_parts(pdf_path: Path, out_dir: Path, pages_per_request: int) -> tuple[str, str, int]:
    """分页段调网关解析，返回 (带页码标记的 md, parse_mode, pdf 总页数)。"""
    src = pymupdf.open(str(pdf_path))
    total_pages = src.page_count
    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    annotated_parts: list[str] = []
    used_fallback = False
    for start in range(0, total_pages, pages_per_request):
        end = min(start + pages_per_request, total_pages)
        sub = pymupdf.open()
        sub.insert_pdf(src, from_page=start, to_page=end - 1)
        part_path = parts_dir / f"part_{start:04d}.pdf"
        sub.save(str(part_path))
        sub.close()
        logger.info(f"解析 {pdf_path.name} 页 {start + 1}-{end}/{total_pages} ...")
        result = colab_parse_pdf(part_path)
        part_path.unlink(missing_ok=True)

        blocks = result.get("content_list")
        if blocks:
            offset_blocks = []
            for block in blocks:
                if isinstance(block, dict):
                    block = dict(block)
                    try:
                        block["page_idx"] = int(block.get("page_idx", 0)) + start
                    except (TypeError, ValueError):
                        pass
                offset_blocks.append(block)
            annotated_parts.append(build_page_annotated_md(offset_blocks))
        else:
            used_fallback = True
            md_text = result.get("md_content") or ""
            annotated_parts.append(
                PAGE_MARKER_TEMPLATE.format(page=start + 1) + "\n\n" + md_text
            )
    src.close()
    mode = "md_fallback" if used_fallback else "content_list"
    return "\n\n".join(annotated_parts), mode, total_pages


def import_one(row: dict, collection: str, pages_per_request: int, force: bool) -> dict:
    stem = Path(row["filename"]).stem
    company = row["company"]
    out_dir = CFQA_OUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "import_summary.json"
    if summary_path.exists() and not force:
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        if prev.get("status") == "ok":
            logger.info(f"跳过已完成: {stem}")
            return prev

    t0 = time.time()
    pdf_path = CFQA_RAW_DIR / row["filename"]
    summary = {
        "stem": stem,
        "company": company,
        "year": row["year"],
        "code": row.get("code", ""),
        "collection": collection,
        "status": "running",
    }
    try:
        md_annotated, parse_mode, total_pages = parse_pdf_in_parts(
            pdf_path, out_dir, pages_per_request
        )
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(md_annotated, encoding="utf-8")
        summary["pdf_pages"] = total_pages
        summary["parse_mode"] = parse_mode
        summary["md_chars"] = len(md_annotated)

        split_node = NodeDocumentSplit()
        res = split_node.process({"md_path": str(md_path), "file_title": stem})
        chunks = res["chunk_dict_list"]
        for chunk in chunks:
            chunk["item_name"] = company
        summary["chunks"] = len(chunks)
        summary["chunks_with_page"] = sum(1 for c in chunks if c.get("page"))

        embed_node = NodeBGEEmbedding()
        embed_node.process({"chunk_dict_list": chunks, "md_path": str(md_path)})

        import_node = NodeImportMilvus()
        import_node.process({"chunk_dict_list": chunks})
        summary["inserted"] = len(chunks)
        summary["status"] = "ok"
    except Exception as e:  # 单份失败不中断批量，错误记入汇总
        summary["status"] = "failed"
        summary["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"{stem} 导入失败: {e}")
    summary["seconds"] = round(time.time() - t0, 1)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{stem}: {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CFQA 年报集导入（Colab 后端）")
    parser.add_argument("--only", help="只导一份，填文件名 stem，如 cfqa_688599_2022")
    parser.add_argument("--all", action="store_true", help="导 manifest 中全部 ok 的年报")
    parser.add_argument("--collection", default="cfqa_colab", help="目标 Milvus 集合名")
    parser.add_argument("--pages-per-request", type=int, default=40)
    parser.add_argument("--skip-done", action="store_true", help="跳过已有 ok 汇总的文档（默认行为即跳过，此开关仅为语义明确）")
    parser.add_argument("--force", action="store_true", help="强制重导已完成的文档")
    args = parser.parse_args()
    if not args.only and not args.all:
        parser.error("必须指定 --only 或 --all")

    # 后端切换：只改本进程内的 config 属性，不动 .env 与主流程默认值
    EmbeddingConfig.embed_source = "colab"
    MineruConfig.parse_backend = "colab"
    MilvusConfig.chunks_collection = args.collection

    health = colab_health()
    logger.info(f"网关健康: {health}")
    if any(v != "ok" for v in health.values()):
        raise RuntimeError(f"网关有服务不健康: {health}")

    rows = load_manifest(CFQA_RAW_DIR / "cfqa_manifest.csv")
    if args.only:
        rows = [r for r in rows if Path(r["filename"]).stem == args.only]
        if not rows:
            raise SystemExit(f"manifest 中找不到: {args.only}")
    logger.info(f"待处理文档: {len(rows)} 份，目标集合: {args.collection}")

    results = [
        import_one(r, args.collection, args.pages_per_request, force=args.force)
        for r in rows
    ]
    ok = sum(1 for s in results if s["status"] == "ok")
    total_chunks = sum(s.get("inserted", 0) for s in results)
    print(f"== 完成: {ok}/{len(results)} 份成功，入库 chunk 合计 {total_chunks}")
    for s in results:
        if s["status"] != "ok":
            print(f"== 失败: {s['stem']}: {s.get('error')}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

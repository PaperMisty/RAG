"""CFQA 检索验证（体验版，非正式评测）。

用 CFQA 金标问题在 cfqa_colab 集合上跑混合检索（dense+sparse，
WeightedRanker 0.8/0.2，与查询流程一致），核对 top-k 命中 chunk 的
page 是否落在金标页集合内，给出 Hit@k 汇总。

口径说明（务必如实看待）：
- 检索按 file_title 过滤到金标问题所属的那一份年报内（文档内检索），
  跨文档区分度不在本次体验范围内；
- 样本量小（默认 10 题，轮选自不同公司），只验证全流程通断与页码
  元数据可用性，不构成评测结论；
- 问题向量走 Colab 网关（EMBED_SOURCE=colab），与入库向量同源。

金标子集 cfqa_gold_subset.json 放在 RAW_DIR/cfqa/ 下（由 CFQA 官方
test 集按 公司+问题年份 匹配 31 份已落地年报、剔除空证据后生成）。

用法：
    .venv/bin/python -m atguigu.import_process.cfqa_validate --limit 10 --k 5
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from atguigu.config.config import OUTPUT_DIR, RAW_DIR, EmbeddingConfig
from atguigu.tool.bgem3_tool import get_bgem3_embedding
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import create_reqs, weighted_hybrid_search

CFQA_RAW_DIR = RAW_DIR / "cfqa"
CFQA_OUT_DIR = OUTPUT_DIR / "cfqa"


def ingested_docs() -> dict[tuple[str, str], str]:
    """扫各文档 import_summary，返回 {(company, year): stem}（仅 status=ok）。"""
    docs = {}
    if CFQA_OUT_DIR.exists():
        for summary_path in CFQA_OUT_DIR.glob("*/import_summary.json"):
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            if s.get("status") == "ok":
                docs[(s["company"], str(s["year"]))] = s["stem"]
    return docs


def pick_questions(gold: list[dict], docs: dict, limit: int) -> list[dict]:
    """在已入库文档的问题里按公司轮选，保证样本跨公司分布。"""
    by_company: dict[str, list[dict]] = defaultdict(list)
    for rec in gold:
        if (rec["company"], str(rec["year"])) in docs:
            by_company[rec["company"]].append(rec)
    picked: list[dict] = []
    companies = sorted(by_company)
    while len(picked) < limit and any(by_company[c] for c in companies):
        for c in companies:
            if by_company[c] and len(picked) < limit:
                picked.append(by_company[c].pop(0))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="CFQA 检索验证（体验版）")
    parser.add_argument("--collection", default="cfqa_colab")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gold", default=str(CFQA_RAW_DIR / "cfqa_gold_subset.json"))
    args = parser.parse_args()

    EmbeddingConfig.embed_source = "colab"

    docs = ingested_docs()
    if not docs:
        raise SystemExit("没有已入库文档（import_summary 无 ok 记录），先跑 cfqa_import")
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    questions = pick_questions(gold, docs, args.limit)
    logger.info(f"已入库 {len(docs)} 份，验证问题 {len(questions)} 题，k={args.k}")

    hits = 0
    details = []
    for rec in questions:
        stem = docs[(rec["company"], str(rec["year"]))]
        gold_pages = set(rec["gold_pages"])
        vecs = get_bgem3_embedding([rec["question"]])
        reqs = create_reqs(
            dense_data=vecs["dense"][0],
            sparse_data=vecs["sparse"][0],
            dense_ann_field="dense",
            sparse_ann_field="sparse",
            limit=args.k,
            expr=f"file_title == '{stem}'",
        )
        res = weighted_hybrid_search(
            collection_name=args.collection,
            reqs=reqs,
            ranker=[0.8, 0.2],
            limit=args.k,
            output_fields=["page", "file_title", "section_title", "chunk_content"],
        )
        top_pages = [hit["entity"].get("page") for hit in res[0]]
        hit = any(p in gold_pages for p in top_pages)
        hits += hit
        row = {
            "id": rec["id"],
            "company": rec["company"],
            "year": rec["year"],
            "question": rec["question"],
            "gold_pages": sorted(gold_pages),
            "top_pages": top_pages,
            "hit": hit,
        }
        details.append(row)
        print(
            f"[{'HIT ' if hit else 'MISS'}] {rec['company']} {rec['year']} "
            f"Q: {rec['question'][:40]} | gold={sorted(gold_pages)} top{args.k}={top_pages}"
        )

    rate = hits / len(questions) if questions else 0
    print(f"== Hit@{args.k}: {hits}/{len(questions)} = {rate:.2f}（文档内检索，体验样本）")
    out_path = CFQA_OUT_DIR / "validate_result.json"
    out_path.write_text(
        json.dumps(
            {"collection": args.collection, "k": args.k, "hit_rate": rate, "details": details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"验证明细写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

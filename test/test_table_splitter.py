"""表格 chunk 行拆分离线单测（用户 2026-10-23 规格）。

只测 atguigu/import_process/table_splitter.py 的纯函数行为：
按完整行拆、每片 <=300 字、结构闭合、表头在每片重复、数据行
不丢不重。不依赖 Milvus/模型/网络。
"""

import re

from atguigu.import_process.table_splitter import expand_chunk_dict, split_table_content

_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TH_RE = re.compile(r"<th\b", re.IGNORECASE)


def build_html_table(n_rows: int = 60) -> tuple[str, list[str], list[str]]:
    header = "<tr><th>科目编号</th><th>科目名称</th><th>期末余额（元）</th></tr>"
    rows = [
        f"<tr><td>KM{i:03d}</td><td>第{i:03d}项长期资产明细科目</td><td>{1_000_000 + i * 137}.00</td></tr>"
        for i in range(1, n_rows + 1)
    ]
    return "<table>" + header + "".join(rows) + "</table>", [header], rows


def build_md_table(n_rows: int = 90) -> tuple[str, str, str, list[str]]:
    header = "| 科目编号 | 科目名称 | 期末余额（元） |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| KM{i:03d} | 第{i:03d}项长期资产明细科目 | {1_000_000 + i * 137}.00 |"
        for i in range(1, n_rows + 1)
    ]
    return "\n".join([header, sep] + rows), header, sep, rows


def test_html_long_table_split_by_rows():
    table, header_rows, data_rows = build_html_table()
    assert len(table) > 3000, f"构造表太短: {len(table)}"
    frags = split_table_content(table)
    assert len(frags) > 1
    header = header_rows[0]
    seen_data_rows: list[str] = []
    for frag in frags:
        assert len(frag) <= 300, f"片段超限: {len(frag)}"
        assert frag.startswith("<table"), frag[:30]
        assert frag.endswith("</table>"), frag[-30:]
        assert header in frag, "每片必须带表头"
        frag_rows = _TR_RE.findall(frag)
        for r in frag_rows:
            assert r.startswith("<tr") and r.endswith("</tr>")
            if not _TH_RE.search(r):
                seen_data_rows.append(r)
    # 数据行不丢不重、顺序一致
    assert seen_data_rows == data_rows


def test_md_long_table_split_by_rows():
    table, header, sep, data_rows = build_md_table()
    assert len(table) > 3000, f"构造表太短: {len(table)}"
    frags = split_table_content(table)
    assert len(frags) > 1
    seen: list[str] = []
    for frag in frags:
        assert len(frag) <= 300, f"片段超限: {len(frag)}"
        lines = frag.split("\n")
        assert lines[0] == header, "每片首行必须是表头行"
        assert lines[1] == sep, "每片第二行必须是分隔行"
        seen.extend(lines[2:])
    assert seen == data_rows


def test_small_table_unchanged():
    table, _, _ = build_html_table(3)
    assert len(table) < 300
    assert split_table_content(table) == [table]


def test_no_table_unchanged():
    text = "这是一段没有表格的长文本。" * 60
    assert len(text) > 300
    assert split_table_content(text) == [text]


def test_title_prefix_kept_with_first_fragment():
    table, header_rows, data_rows = build_html_table()
    content = "## 固定资产明细表\n" + table
    frags = split_table_content(content)
    assert len(frags) > 1
    assert all(len(f) <= 300 for f in frags)
    assert frags[0].startswith("## 固定资产明细表"), frags[0][:40]
    seen: list[str] = []
    for frag in frags:
        for r in _TR_RE.findall(frag):
            if not _TH_RE.search(r):
                seen.append(r)
    assert seen == data_rows


def test_single_huge_row_not_broken():
    huge_row = "<tr><td>" + "长" * 400 + "</td></tr>"
    header = "<tr><th>列</th></tr>"
    table = "<table>" + header + huge_row + "</table>"
    frags = split_table_content(table)
    joined = "".join(frags)
    assert huge_row in joined, "超长单行必须整行保留、不许硬切"
    assert joined.count(huge_row) == 1


def test_expand_chunk_dict_inherits_metadata():
    chunk = {
        "file_title": "测试文档",
        "section_title": "## 表",
        "chunk_content": "x",
        "part": 0,
        "page": 7,
        "item_name": "测试主体",
        "dense": [0.1],
        "sparse": {1: 0.5},
        "id": 42,
    }
    out = expand_chunk_dict(chunk, ["片一", "片二", "片三"])
    assert [c["part"] for c in out] == [1, 2, 3]
    for c in out:
        assert c["file_title"] == "测试文档"
        assert c["section_title"] == "## 表"
        assert c["page"] == 7
        assert c["item_name"] == "测试主体"
        assert "dense" not in c and "sparse" not in c and "id" not in c
    # 未拆分原样返回（同一对象、向量保留）
    same = expand_chunk_dict(chunk, ["x"])
    assert same == [chunk] and same[0] is chunk

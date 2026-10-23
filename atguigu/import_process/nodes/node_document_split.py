# atguigu/import_process/nodes/node_document_split.py
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from langchain_text_splitters import RecursiveCharacterTextSplitter
from atguigu.import_process import state
from pathlib import Path
from atguigu.tool.validate_path import validate_path
from atguigu.config.config import OUTPUT_DIR
from atguigu.tool.logger import logger
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.colab_client import PAGE_MARKER_RE
from atguigu.import_process.table_splitter import split_table_content
import re


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：父子文档切分方式,先按段落(父),再按换行符和标点符号(子)
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # 1.获取Markdown文本内容,以换行符切分为列表
        md_content_list, file_title, md_path_obj = self.get_md_content(state)
        # 获取是否为 PPT 的标记
        is_ppt = state.get("is_ppt_read_enabled", False)
        # 2.获取Markdown每个章节内容,将每个章节切分为chunks
        chunk_dict_list = self.split_section(md_content_list, file_title, is_ppt)
        # 3.保存
        self.save_chunks_json(md_path_obj, chunk_dict_list)
        # 4.返回chunks详细信息
        return {"chunk_dict_list": chunk_dict_list}

    def get_md_content(self, state: ImportGraphState) -> tuple[list[str], str, Path]:
        """读取Markdown文件, 做换行标准化，按换行切割返回文本行列表。

        Args:
            state (ImportGraphState): LangGraph导入流程状态对象
                - md_path: markdown文件路径
                - file_title: 可选，文件标题；不存在则使用文件名stem

        Raises:
            Exception: 当md文件读取后内容为空字符串时抛出；路径校验失败由validate_path内部抛出。

        Returns:
            list[str]: markdown按换行分割后的每行文本组成的列表，不含原始\r\n换行符。
            file_title: 有效的文件名
        """
        md_path_obj = Path(state.get("md_path", ""))
        md_path_obj = validate_path(md_path_obj, "error")
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        if not md_content:
            logger.debug(f"{md_path_obj}无内容")
            return {}

        file_title = state.get("file_title")
        if not file_title:
            file_title = md_path_obj.stem

        # 统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\n\n", "\n")
        md_content_list = md_content.split("\n")
        return md_content_list, file_title, md_path_obj

    def split_section(self, md_content_list: list[str], file_title: str, is_ppt: bool = False) -> list[dict]:
        """按 # 字符来切分段落

        页码处理（CFQA/Colab 接入）：若 md 带有 <!-- page:N --> 页码标记
        （由 colab_client.build_page_annotated_md 生成，N 为 1 起 PDF 页序），
        先剥离标记并记录每行页码；切分时除标题边界外，页边界也作为切分点，
        使每个 chunk 的 page 精确到页（同标题跨页的续段沿用该标题）。
        无标记的旧文档行为与原来完全一致，所有 chunk 的 page 为 0（未知）。

        Args:
            md_content_list (list[str]): Markdown文本列表
            file_title (str): 文件名称
            is_ppt (bool): 是否是 PPT 幻灯片路径

        Returns:
            list[dict]: 切分后的段落信息（每条带 page 字段）
        """
        # ---- 页码预处理：剥离标记行，得到干净行 + 逐行页码 ----
        clean_lines: list[str] = []
        line_pages: list[int] = []
        page = 0
        for raw_line in md_content_list:
            marker = PAGE_MARKER_RE.match(raw_line.strip())
            if marker:
                page = int(marker.group(1))
                continue
            clean_lines.append(raw_line)
            line_pages.append(page)

        is_block = False
        code_fence = None
        code_pattern = r"^(`{3,}|~{3,})"
        title_pattern = r"^(#{1,6})\s+.*"

        current_idx = 0
        # 当前累积段不以 # 开头时使用的标题：文档开头为 "摘要"；
        # 某标题段结算后更新为该标题，供其跨页续段沿用（父子定位语义不变）
        pending_title = "摘要"
        chunk_dict_list = []

        def flush(end_idx: int) -> None:
            nonlocal current_idx, pending_title
            section_list = clean_lines[current_idx:end_idx]
            if section_list:
                section_content = "\n".join(section_list)
                if section_content.startswith("#"):
                    section_title = section_list[0]
                    pending_title = section_title
                else:
                    section_title = pending_title
                section_page = line_pages[current_idx] if current_idx < len(line_pages) else 0
                _chunk_dict_list = self.split_chunks(
                    file_title, section_title, section_content, is_ppt, page=section_page
                )
                chunk_dict_list.extend(_chunk_dict_list)
            current_idx = end_idx

        total = len(clean_lines)
        for idx, raw_line in enumerate(clean_lines):
            line = raw_line.strip()
            code_matched = re.match(code_pattern, line)
            # 判断是否处于代码围栏内
            if code_matched:
                if not is_block:
                    is_block = True
                    code_fence = code_matched.group(1)
                elif code_fence == code_matched.group(1):
                    is_block = False
                    code_fence = None

            if is_block:
                continue
            is_last = idx == total - 1
            page_changed = idx > current_idx and line_pages[idx] != line_pages[current_idx]
            # 触发结算的优先级与原逻辑对齐：最后一行 > 标题行 > 页边界
            if is_last:
                flush(idx + 1)
            elif re.match(title_pattern, line) or page_changed:
                flush(idx)
        return chunk_dict_list

    def split_chunks(
        self,
        file_title: str,
        section_title: str,
        section_content: str,
        is_ppt: bool = False,
        page: int = 0,
    ) -> list[dict]:
        """递归切割器切分段落,给每个Chunk分配段落标题,
        对于含有HTML的和小于MAX_LENGTH的暂不切分

        Args:
            section_content (str): 段落信息
            file_title (str): 文件名
            is_ppt (bool): 是否是 PPT 幻灯片路径
            page (int): 本段所属页（1 起 PDF 页序；0 表示未知，旧文档恒为 0）

        Returns:
            list[dict]: 切分后的 chunk 字典列表
        """
        MAX_LENGTH = 300
        CHUNK_OVERLAP = 30
        chunk_dict_list = []
        spliter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            chunk_size=MAX_LENGTH,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # PPT 幻灯片特有逻辑：提取该 Section 中引用的图片 Markdown 标签
        image_md = ""
        if is_ppt:
            # 正则匹配类似于 ![DK123405-1](http://...) 的图片标签
            image_match = re.search(r"!\[.*?\]\(.*?\)", section_content)
            if image_match:
                image_md = image_match.group(0)

        # TODO 去除段落标题应该在最前面进行
        is_table = ("<table" in section_content) or (re.search(r"\|\s*[-:]+\s*\|", section_content) is not None)
        if len(section_content) < 300 or is_table:
            chunk_contents = [section_content]
            if is_table and len(section_content) >= MAX_LENGTH:
                # 表格行拆分（用户 2026-10-23 规格）：含表格的 section 原本
                # 整段成 chunk，超长表会同时踩 Milvus varchar 硬报错与
                # BGE-M3 静默截断两层坑；此处按完整行拆成 <=300 字的
                # 结构完整迷你表（见 import_process/table_splitter）。
                # TODO(长表回归): 当前语料无极端长表文档，真长表出现时需复测。
                chunk_contents = split_table_content(section_content, MAX_LENGTH)
            # 未拆分时 part=0（与原行为一致）；拆分时片段按 1..N 顺延编号
            part_numbers = [0] if len(chunk_contents) == 1 else range(1, len(chunk_contents) + 1)
            for chunk_content, part_no in zip(chunk_contents, part_numbers):
                chunk_dict_list.append(
                    {
                        "file_title": file_title,
                        "section_title": section_title,  # section_title作用在于父子文档切分时候的定位, chunk进行向量化不附件section_title, 而是附加item_name
                        "chunk_content": chunk_content,
                        "part": part_no,
                        "page": page,
                    }
                )
        else:
            real_section_content = section_content[len(section_title) :]  # 去除段落标题
            chunk_list = spliter.split_text(real_section_content)
            for idx, chunk in enumerate(chunk_list, 1):
                chunk_content = section_title + "\n\n" + chunk
                # PPT 幻灯片特有逻辑：若切出来的子 chunk 内不含图片，则在尾部附加上图片线上 URL
                if is_ppt and image_md and image_md not in chunk_content:
                    chunk_content = chunk_content + "\n\n" + image_md
                chunk_dict_list.append(
                    {
                        "file_title": file_title,
                        "section_title": section_title,
                        "chunk_content": chunk_content,
                        "part": idx,
                        "page": page,
                    }
                )
        return chunk_dict_list

    def save_chunks_json(self, md_path_obj: Path, chunk_dict_list: list[dict]) -> None:
        """保存Markdown文档已切分的chunks信息

        Args:
            md_path_obj (Path):
            chunk_dict_list (list[dict]):
        """
        chunks_json_path = str(md_path_obj.parent) + "/" + md_path_obj.stem + "_chunks.json"
        with open(
            chunks_json_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json_format(chunk_dict_list))
        logger.info(f"{chunks_json_path}文件成功写入")


if __name__ == "__main__":
    node = NodeDocumentSplit()
    init_state = {
        "md_path": str(OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new.md"),
        "file_title": "hak180产品安全手册",
    }

    res = node(init_state)
    logger.info(json_format(res))

# atguigu/import_process/nodes/node_ppt_to_md.py
import asyncio
import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import timedelta
from pathlib import Path

try:
    import pythoncom
    import win32com.client
except ImportError:
    pythoncom = None
    win32com = None

import requests
import urllib3
from PIL import Image

from atguigu.config.config import MinIOConfig, MineruConfig, OUTPUT_DIR, RAW_DIR
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger
from atguigu.tool.minio_client_tool import get_minio_client
from atguigu.tool.validate_path import validate_path


class NodePPTToMD(NodeBase):
    """
    PPT 转 Markdown 节点：
    1. 将 PPT/PPTX 各页 Slide 转为图片 (通过 win32com 驱动 PowerPoint 或 WPS)
    2. 等比例限制最长边在 1024-2048 像素之间
    3. 上传图片至 MinIO，支持获取匿名直连 URL 或临时预签名 URL
    4. 调用 MinerU 在线 API 并发异步批处理解析每页 Slide 提取 Markdown 文本
    5. 组装成最终的 Markdown 文档
    """

    name = "node_ppt_to_md"

    def __init__(self, use_presigned_url: bool = False):
        """
        Args:
            use_presigned_url (bool): 是否使用 MinIO 的预签名 URL 写入 Markdown。默认 False。
        """
        self.use_presigned_url = use_presigned_url
        self.token = MineruConfig.mineru_token
        self.base_url = MineruConfig.mineru_base_url
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 校验输入路径
        ppt_path_str = state.get("ppt_path")
        local_dir_str = state.get("local_dir")

        if not ppt_path_str or not local_dir_str:
            raise ValueError("ppt_path 或 local_dir 不能为空")

        ppt_path = Path(ppt_path_str)
        local_dir = Path(local_dir_str)

        if not ppt_path.exists():
            raise FileNotFoundError(f"未找到指定的 PPT 文件: {ppt_path}")

        local_dir.mkdir(parents=True, exist_ok=True)
        file_title = state.get("file_title", ppt_path.stem)

        # 临时工作目录，用来存放 PPT 导出的原始图片与缩放后的图片
        temp_work_dir = local_dir / f"temp_{file_title}"
        if temp_work_dir.exists():
            shutil.rmtree(temp_work_dir)
        temp_work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: PPT 导出图片
            raw_export_dir = temp_work_dir / "raw_slides"
            raw_export_dir.mkdir(parents=True, exist_ok=True)
            self._export_ppt_to_images(ppt_path, raw_export_dir)

            # 获取并按页码排序导出的图片
            raw_images = [f for f in os.listdir(raw_export_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not raw_images:
                raise RuntimeError("PPT 导出图片失败，未在导出目录中找到任何图片文件。")

            # 按照文件名中的数字进行排序，避免字符排序问题 (如 Slide10.JPG 排在 Slide2.JPG 前面)
            def get_slide_num(filename):
                nums = re.findall(r"\d+", filename)
                return int(nums[0]) if nums else 0

            raw_images.sort(key=get_slide_num)
            logger.info(f"成功导出 PPT，共有 {len(raw_images)} 页 Slide。")

            # 确定最终 Markdown 及其图片的输出目录
            md_dir = local_dir / file_title
            md_dir.mkdir(parents=True, exist_ok=True)
            target_img_dir = md_dir / "images"
            target_img_dir.mkdir(parents=True, exist_ok=True)

            # Step 2: 尺寸缩放，并直接保存到本地的 images 目录下
            resized_img_paths = []
            img_local_urls = []

            for idx, img_name in enumerate(raw_images, 1):
                raw_img_path = raw_export_dir / img_name
                resized_img_path = target_img_dir / f"{file_title}-{idx}.jpg"

                # 等比例限制最长边在 1024-2048 像素之间，输出至本地 images 文件夹下
                self._resize_image(raw_img_path, resized_img_path)
                resized_img_paths.append(resized_img_path)

                # 使用本地相对路径作为生成的 Markdown 图片引用地址
                img_local_url = f"images/{file_title}-{idx}.jpg"
                img_local_urls.append(img_local_url)

            # Step 4: 异步并发调用 MinerU 在线 API 解析所有 Slide 图片，内插图同样解压至 target_img_dir
            logger.info("开始并发调用 MinerU 解析所有 Slide 图片...")
            mineru_markdowns = asyncio.run(self._extract_slides_content_mineru_async(resized_img_paths, target_img_dir))

            # Step 5: 拼装最终 Slide 的 Markdown 内容（使用相对路径，交由下游 NodeMDImg 统一 VLM 总结和替换）
            markdown_sections = []
            for idx, (img_local_url, md_content) in enumerate(zip(img_local_urls, mineru_markdowns), 1):
                title = self._extract_title_from_md(md_content)
                slide_section = (
                    f"# Slide {idx}: {title}\n\n" f"![{file_title}-{idx}]({img_local_url})\n\n" f"{md_content}\n"
                )
                markdown_sections.append(slide_section)

            # 拼接完整的 Markdown 全文
            md_content = f"# {file_title}\n\n" + "\n---\n\n".join(markdown_sections)

            # 将组装后的 Markdown 写入磁盘
            target_md_path = md_dir / f"{file_title}.md"
            target_md_path.write_text(md_content, encoding="utf-8")
            logger.info(f"Markdown 文件已生成: {target_md_path}")

            return {
                "md_content": md_content,
                "md_path": str(target_md_path),
            }

        finally:
            # 统一清理临时工作目录
            if temp_work_dir.exists():
                shutil.rmtree(temp_work_dir)
                logger.info("已清理临时 PPT 图片转换目录。")

    def _export_ppt_to_images(self, ppt_path: Path, output_dir: Path) -> None:
        """调用 Windows PowerPoint/WPS 导出 PPT 为 JPG 图片"""
        if pythoncom is None or win32com is None:
            raise NotImplementedError("当前非 Windows 环境，缺少 win32com/pythoncom 支持，无法执行本地 PPT 转图片。")
        pythoncom.CoInitialize()
        ppt_app = None
        presentation = None
        try:
            # 1. 尝试使用 Microsoft PowerPoint
            try:
                ppt_app = win32com.client.Dispatch("PowerPoint.Application")
            except Exception as e:
                logger.warning(f"无法启动 PowerPoint.Application，尝试 WPS: {e}")
                # 2. 尝试使用 WPS WPP 演示
                try:
                    ppt_app = win32com.client.Dispatch("Wpp.Application")
                except Exception as e_wps:
                    logger.error(f"无法启动 WPS Wpp.Application: {e_wps}")
                    raise RuntimeError("未检测到系统安装有 PowerPoint 或 WPS Office，无法导出 PPT 图片。")

            ppt_path_abs = str(ppt_path.resolve())
            output_dir_abs = str(output_dir.resolve())

            # Open 核心参数：Open(FileName, ReadOnly, Untitled, WithWindow)
            # WithWindow=False 代表在后台无窗口运行
            presentation = ppt_app.Presentations.Open(ppt_path_abs, ReadOnly=True, Untitled=False, WithWindow=False)
            # SaveAs(FileName, FileFormat): 17 代表 ppSaveAsJPG
            presentation.SaveAs(output_dir_abs, 17)
        finally:
            if presentation:
                try:
                    presentation.Close()
                except Exception:
                    pass
            if ppt_app:
                try:
                    ppt_app.Quit()
                except Exception:
                    pass
            pythoncom.CoUninitialize()

    def _resize_image(self, src_path: Path, dest_path: Path) -> None:
        """限制最长边在 1024-2048 像素之间，等比例缩放"""
        with Image.open(src_path) as img:
            width, height = img.size
            max_side = max(width, height)

            new_width, new_height = width, height
            if max_side > 2048:
                scale = 2048 / max_side
                new_width, new_height = int(width * scale), int(height * scale)
            elif max_side < 1024:
                scale = 1024 / max_side
                new_width, new_height = int(width * scale), int(height * scale)

            if (new_width, new_height) != (width, height):
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"图片缩放: 原尺寸 ({width}x{height}) -> 新尺寸 ({new_width}x{new_height})")

            # 强制转换为 RGB 并以 JPG 保存
            img.convert("RGB").save(dest_path, "JPEG", quality=90)

    def _clear_minio_ppt_dir(self, minio_client, file_title: str) -> None:
        """清理 MinIO 中已存在的同名 PPT 图片"""
        from minio.deleteobjects import DeleteObject

        prefix = f"{MinIOConfig.minio_img_dir}/{file_title}"
        try:
            objects_to_delete = minio_client.list_objects(
                bucket_name=MinIOConfig.minio_bucket_name, prefix=prefix, recursive=True
            )
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                errors = minio_client.remove_objects(
                    bucket_name=MinIOConfig.minio_bucket_name, delete_object_list=delete_list
                )
                for err in errors:
                    logger.error(f"删除 MinIO 历史图片失败: {err}")
        except Exception as e:
            logger.warning(f"清理 MinIO 历史 PPT 目录时发生异常: {e}")

    def _extract_title_from_md(self, md_content: str) -> str:
        """从 Markdown 文本中智能抓取合适的一级标题作为页名"""
        # 优先正则查找一级标题 #, ## 等
        match = re.search(r"^#+\s+(.+)$", md_content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # 退化逻辑：抓取第一行非空、且不是图片引用的纯文字
        for line in md_content.splitlines():
            line = line.strip()
            if line and not line.startswith("!") and not line.startswith("["):
                return line[:30]  # 长度截断
        return ""

    async def _extract_slides_content_mineru_async(self, img_paths: list[Path], target_img_dir: Path) -> list[str]:
        """使用 MinerU 批量上传与异步并发拉取解析结果"""
        if not self.token or not self.base_url:
            raise ValueError("MineruConfig.mineru_token 或 mineru_base_url 未正确配置。")

        # 1. 批量申请直传 URL
        apply_url = f"{self.base_url}/file-urls/batch"
        files_payload = [{"name": path.name, "data_id": str(idx)} for idx, path in enumerate(img_paths, 1)]
        payload = {
            "files": files_payload,
            "model_version": "vlm",
        }

        logger.info("正在向 MinerU 申请批量直传链接...")
        res = requests.post(apply_url, headers=self.header, json=payload)
        if res.status_code != 200 or res.json().get("code") != 0:
            raise RuntimeError(f"申请 MinerU 上传文件链接失败: {res.text}")

        result_data = res.json()["data"]
        batch_id = result_data["batch_id"]
        upload_urls = result_data["file_urls"]
        logger.info(f"批量申请成功. batch_id: {batch_id}, 获得 {len(upload_urls)} 个直传链接。")

        # 2. 异步并行上传所有 Slide 图片
        def upload_single_file(url: str, file_path: Path):
            logger.info(f"正在直传文件至 MinerU: {file_path.name}")
            with open(file_path, "rb") as f:
                res_upload = requests.put(url, data=f)
                if res_upload.status_code != 200:
                    raise RuntimeError(f"直传文件 {file_path.name} 失败，S3状态码: {res_upload.status_code}")
            logger.info(f"文件直传成功: {file_path.name}")

        upload_tasks = [
            asyncio.to_thread(upload_single_file, u_url, img_p) for u_url, img_p in zip(upload_urls, img_paths)
        ]
        await asyncio.gather(*upload_tasks)
        logger.info("所有 Slide 图片均已直传至 MinerU，开始进入解析阶段...")

        # 3. 异步并发轮询结果包状态
        poll_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        extract_results = []

        while True:

            def fetch_status():
                return requests.get(poll_url, headers=self.header)

            status_res = await asyncio.to_thread(fetch_status)
            if status_res.status_code != 200 or status_res.json().get("code") != 0:
                raise RuntimeError(f"轮询解析状态失败: {status_res.text}")

            results_data = status_res.json()["data"]["extract_result"]
            all_done = all(item["state"] in ("done", "failed") for item in results_data)

            if all_done:
                extract_results = results_data
                break

            logger.info("MinerU 在线解析进行中，请耐心等待...")
            await asyncio.sleep(2)

        logger.info("MinerU 在线解析完成，开始并发下载并读取解析文本...")

        # 4. 异步并发下载结果 ZIP 包，支持防代理拦截下载 (带 curl 退化兜底)
        def download_zip_bytes(zip_url: str, data_id: str) -> bytes:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            zip_content = None

            try:
                # 策略1：绕过代理下载
                res_zip = requests.get(zip_url, verify=False, proxies={"http": None, "https": None}, timeout=30)
                if res_zip.status_code == 200:
                    zip_content = res_zip.content
                else:
                    logger.warning(
                        f"[Slide {data_id}] Requests 下载失败(状态码: {res_zip.status_code})，尝试系统 curl..."
                    )
            except Exception as e:
                logger.warning(f"[Slide {data_id}] Requests 下载异常: {e}，尝试系统 curl...")

            if zip_content is None:
                # 策略2：调用系统 curl 直传/下载
                temp_dir = tempfile.gettempdir()
                temp_zip_path = os.path.join(temp_dir, f"mineru_ppt_{batch_id}_{data_id}.zip")
                try:
                    logger.info(f"[Slide {data_id}] 正在通过系统 curl 进程拉取 ZIP 文件...")
                    res = subprocess.run(["curl", "-L", "-o", temp_zip_path, zip_url], capture_output=True, text=True)
                    if res.returncode == 0 and os.path.exists(temp_zip_path) and os.path.getsize(temp_zip_path) > 0:
                        with open(temp_zip_path, "rb") as f:
                            zip_content = f.read()
                        os.remove(temp_zip_path)
                        logger.info(f"[Slide {data_id}] 系统 curl 进程下载成功！")
                    else:
                        logger.error(
                            f"[Slide {data_id}] curl 进程下载失败，返回码: {res.returncode}, 错误信息: {res.stderr}"
                        )
                except Exception as curl_err:
                    logger.error(f"[Slide {data_id}] 调用系统 curl 发生错误: {curl_err}")

            if zip_content is None:
                raise ValueError(
                    f"下载 [Slide {data_id}] 解析包失败，Requests 与 Curl 均无法获取文件。请检查代理设置。"
                )

            return zip_content

        def extract_md_and_images_from_zip(zip_content: bytes, img_out_dir: Path) -> str:
            md_text = ""
            with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zip_ref:
                # 检查是否存在图片，若有则创建 images 文件夹
                has_images = any("images/" in name and not name.endswith("/") for name in zip_ref.namelist())
                if has_images:
                    img_out_dir.mkdir(parents=True, exist_ok=True)

                for member in zip_ref.infolist():
                    if member.filename.endswith(".md"):
                        with zip_ref.open(member) as f:
                            md_text = f.read().decode("utf-8")
                    elif "images/" in member.filename and not member.is_dir():
                        base_name = os.path.basename(member.filename)
                        if base_name:
                            out_path = img_out_dir / base_name
                            with zip_ref.open(member) as source, open(out_path, "wb") as target:
                                shutil.copyfileobj(source, target)
            return md_text

        async def download_and_extract_task(item):
            data_id = item["data_id"]
            state = item["state"]
            if state != "done":
                err_msg = item.get("err_msg", "Unknown error")
                logger.error(f"[Slide {data_id}] MinerU 解析失败: {err_msg}")
                return int(data_id), f"\n> [!WARNING]\n> 本页 Slide 解析失败，错误信息: {err_msg}\n"

            zip_url = item["full_zip_url"]
            # 运行于线程池防止 IO 阻塞
            zip_bytes = await asyncio.to_thread(download_zip_bytes, zip_url, data_id)
            md_content = extract_md_and_images_from_zip(zip_bytes, target_img_dir)
            return int(data_id), md_content

        download_tasks = [download_and_extract_task(item) for item in extract_results]
        downloaded_contents = await asyncio.gather(*download_tasks)

        # 按照 data_id 重新排序以确保 PPT 图片页的顺序一致
        downloaded_contents.sort(key=lambda x: x[0])

        return [content for _, content in downloaded_contents]


if __name__ == "__main__":
    # 单节点调试
    import logging

    logging.basicConfig(level=logging.INFO)

    node = NodePPTToMD(use_presigned_url=False)

    # 查找 raw 目录中的真实 PPTX 文件进行单节点测试
    raw_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pptx") and not f.startswith("~$")]
    if not raw_files:
        logger.info("未找到测试 PPT 文件，测试已跳过。")
    else:
        test_ppt_name = raw_files[0]
        test_ppt_path = RAW_DIR / test_ppt_name
        logger.info(f"开始测试，选择测试文件: {test_ppt_path}")

        init_state = {
            "ppt_path": str(test_ppt_path),
            "local_dir": str(OUTPUT_DIR),
            "file_title": test_ppt_path.stem,
        }
        res = node(init_state)
        print("测试返回状态 md_path：", res.get("md_path"))

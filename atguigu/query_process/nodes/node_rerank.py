# atguigu/query_process/nodes/node_rerank.py

# 【Colab远程部署改造】原百炼 DashScope 版 reranker_tool 改为自托管远程版（函数签名与返回结构不变）
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from atguigu.tool.reranker_tool import text_rerank  # 回滚百炼 DashScope 版(改造时保留未删);自托管远程版见 reranker_http_tool,待统一分发后再切换
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger, summarize_docs
from observer.collector import trace_collector
import time


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        task_id = state.get("task_id") or ""
        # 1. 重排序 (记录 Reranker HTTP IO 耗时)
        t_rerank = time.perf_counter()
        rerank_merge_docs = self.get_rerank_docs(state)
        rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 2)

        # 2. 断崖检测 (记录 Cliff CPU 耗时)
        t_cliff = time.perf_counter()
        reranked_docs = self.cliff_detection(rerank_merge_docs)
        cliff_ms = round((time.perf_counter() - t_cliff) * 1000, 2)

        if task_id:
            trace_collector.add_io_item(task_id, "reranker", "Cross-Encoder Rerank HTTP IO", rerank_ms, io_type="network")
            trace_collector.set_module_total_ms(task_id, "reranker", rerank_ms)
            trace_collector.add_io_item(task_id, "cliff", "断崖分差截断 CPU 计算", cliff_ms, io_type="cpu")
            trace_collector.set_module_total_ms(task_id, "cliff", cliff_ms)
            trace_collector.record_rerank_and_cliff(
                task_id=task_id,
                rerank_all_docs=rerank_merge_docs,
                cliff_kept_docs=reranked_docs,
                abs_gap_thres=0.25,
                ratio_gap_thres=0.10,
                min_len=3,
                max_len=10,
            )

        return {"reranked_docs": reranked_docs}

    def cliff_detection(self, rerank_merge_docs: list[dict]) -> list[dict]:
        """固定参数的断崖检测, 绝对差值or相对差值的断崖

        Args:
            doc_lst (list[dict]): _description_

        Returns:
            list[dict]: _description_
        """
        ABS_GAP_THRES = 0.25
        RATIO_GAP_THRES = 0.1
        MIN_LEN = 3
        MAX_LEN = 10
        ACTUAL_MAX_LEN = min(MAX_LEN, len(rerank_merge_docs))
        ACTUAL_MIN_LEN = min(MIN_LEN, len(rerank_merge_docs))
        for i in range(ACTUAL_MIN_LEN - 1, ACTUAL_MAX_LEN - 1):
            current_score = rerank_merge_docs[i]["score"]
            next_score = rerank_merge_docs[i + 1]["score"]
            abs_gap = current_score - next_score
            ratio_gap = abs_gap / current_score
            if abs_gap > ABS_GAP_THRES or ratio_gap > RATIO_GAP_THRES:
                return rerank_merge_docs[: i + 1]
        return rerank_merge_docs[:ACTUAL_MAX_LEN]

    def get_rerank_docs(self, state: QueryGraphState) -> list[dict]:
        """把Web和Local的查询进行精细重排序

        Args:
            state (QueryGraphState): _description_

        Raises:
            ValueError: _description_

        Returns:
            list[dict]: _description_
        """
        rewritten_query = state.get("rewritten_query")
        rrf_chunks = state.get("rrf_chunks")
        web_search_docs = state.get("web_search_docs")

        # web 为补充分支（可能因 MCP 超时返回空），不能因 web 为空就否定已有的 KB 召回。
        # 仅当 KB 与 web 双空（或无重写问题）才算无召回；合并池取各自现有的部分。
        if (not rrf_chunks and not web_search_docs) or not rewritten_query:
            logger.error(f"没有召回结果: {rrf_chunks=},{web_search_docs=},{rewritten_query=}")
            raise ValueError(f"没有召回结果: {rrf_chunks=},{web_search_docs=},{rewritten_query=}")

        merge_docs = (rrf_chunks or []) + (web_search_docs or [])
        cleaned_docs = []
        for doc in merge_docs:
            content_val = doc.get("chunk_content") or doc.get("content")
            if content_val and isinstance(content_val, str):
                cleaned_docs.append({
                    "id": doc.get("id"),
                    "item_name": doc.get("item_name"),
                    "rrf_rank": doc.get("rrf_rank"),
                    "title": doc.get("item_name") or doc.get("title") or "未知标题",
                    "content": content_val,
                    "source": doc.get("source"),
                    "url": doc.get("url"),
                    "file_title": doc.get("file_title"),
                    "section_title": doc.get("section_title"),
                })
        merge_docs = cleaned_docs
        documents = [
            f"{doc.get('file_title') or ''} {doc.get('section_title') or ''}\n{doc.get('content') or ''}".strip()
            for doc in merge_docs
        ]
        # 获取Web和Local重排序的排名。dashscope 调用本身没有 timeout
        # 参数（实测一次跑了 45 秒），直接调会在上游卡住时让整图无限
        # 挂起。120 秒是底线而非目标：正常重排远低于此，超时后图走
        # 统一的 error 事件链路，前端不再静默转圈。注意：超时后 worker
        # 线程仍在后台跑完（无法安全杀掉），只是图不再等它。
        rerank_executor = ThreadPoolExecutor(max_workers=1)
        try:
            rerank_future = rerank_executor.submit(
                text_rerank, query=rewritten_query, documents=documents, top_n=len(merge_docs)
            )
            rerank_lst = rerank_future.result(timeout=120)
        except FutureTimeoutError:
            rerank_executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError("重排序调用超时（120 秒），请重试")
        finally:
            rerank_executor.shutdown(wait=False, cancel_futures=True)
        # 赋值分数
        for item in rerank_lst:
            merge_docs[item["index"]]["score"] = item["score"]
        # 按分数排名
        rerank_merge_docs = sorted(merge_docs, key=lambda x: x["score"], reverse=True)
        # =========== 展示示例 ==========
        logger.info(f"Reranker重排结果: {summarize_docs(rerank_merge_docs, limit=3)}")
        return rerank_merge_docs


if __name__ == "__main__":
    node = NodeRerank()
    init_state = {
        "rewritten_query": "关于HAK180烫金机使用",
        "rrf_chunks": [
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![设备需放置于平稳通风处，避免震动；搬运时双手托底，勿触危险区域；使用后断电，注意纸张边缘锋利。](http://192.168.100.88:9000/knowledge-base/upload-images/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788527,
                "score": 0.830259382724762,
                "source": "local",
                "score_rrf": 0.03252247488101534,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![设备使用需注意防火、防触电，避免儿童接触塑料袋，使用后待冷却再开盖，防止烧伤。](http://192.168.100.88:9000/knowledge-base/upload-images/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788521,
                "score": 0.8251358270645142,
                "source": "local",
                "score_rrf": 0.032266458495966696,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788513,
                "score": 0.8236921429634094,
                "source": "local",
                "score_rrf": 0.03149801587301587,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改 ，或者第三方提出的任何索赔，我们不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788514,
                "score": 0.8140485882759094,
                "source": "local",
                "score_rrf": 0.030776515151515152,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![确保设备放置平稳，远离边缘，使用时勿将手伸入纸张边缘，搬运需双手托底，避免跌落造成伤害或损坏。](http://192.168.100.88:9000/knowledge-base/upload-images/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788530,
                "score": 0.8157209753990173,
                "source": "local",
                "score_rrf": 0.030536130536130537,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![使用起搏器者需远离设备，注意高温部件防烫伤；设备须接220-240V交流电，禁用直流电源，防止触电或火灾。](http://192.168.100.88:9000/knowledge-base/upload-images/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788522,
                "score": 0.8076007962226868,
                "source": "local",
                "score_rrf": 0.029877369007803793,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "无标题",
                "content": "![HAK 180烫金机产品安全手册，含使用前须知、安全提示及获取说明书的官方网址。](http://192.168.100.88:9000/knowledge-base/upload-images/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN\n",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788512,
                "score": 0.8104063868522644,
                "source": "local",
                "score_rrf": 0.029418126757516764,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以 及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788517,
                "score": 0.8088258504867554,
                "source": "local",
                "score_rrf": 0.029411764705882353,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![本设备需接地使用，放置于平稳通风处，避免灰尘堆积和手指误入危险区域，搬运时用双手抓稳。](http://192.168.100.88:9000/knowledge-base/upload-images/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788526,
                "score": 0.8287613987922668,
                "source": "local",
                "score_rrf": 0.016129032258064516,
            },
            {
                "id": 468273558621788532,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合 格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.819697380065918,
                "source": "local",
                "score_rrf": 0.014925373134328358,
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本 设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788516,
                "score": 0.6857563853263855,
                "source": "local",
                "score_rrf": 0.014285714285714285,
            },
            {
                "id": 999999999999999999,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![本设备需接地使用，放置于平稳通风处，避免灰尘堆积和手指误入危险区域，搬运时用双手抓稳。](http://192.168.100.88:9000/knowledge-base/upload-images/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.6666666666666,
                "source": "local",
                "score_rrf": 0.014285714285714285,
            },
        ],
        "web_search_docs": [
            {
                "title": "无版烫金+连续烫印?论一台优秀烫金机的自我修养!兄弟HAK180烫金机评测",
                "content": "去年底,Brother在进博会发布HAK180烫金机,作为Brother旗下新品类,烫金机是其在打印机、一体机、标签机、条码机、扫描仪等之后,布局的又一办公文印设备品。作为一款主要针对高端文印店推出的产品,HAK180的问世,令烫金品在文印店中即可完成,无需再像以前跑到制作工厂去定制,简化流程提升效率;对于烫金需求方 而言,也就是企业、学校、花店等,无需频繁的确认,减少了制作流程,向文印店提出需求后,在文印店中就可完成,简单的烫金需求甚至可以做到“立等可取”,一改了传统需要在“需求方,供应商,制作工厂”间频繁沟通、确认、修改的流程,HAK180让烫金流程更省时、更省力、更省沟通。那么,烫金机究竟如何工作,长相又如何,且随着笔者一同去认识这款产 品! 我们先观看一段视频,了解下烫金机的用途 细分市场需求,灵巧机身,任性安置 近年来,随着文印市场逐渐呈现精细化发展趋势,高端文印的需求逐渐增加,大势之下兄弟HAK180烫金机应运而生。烫金机,顾名思义,可以简单理解为,在纸张表面烫印一层金色,当然,此“金”非彼“金”,就像上面提到的奖状、春联,只是在技术上有些特殊。 第一眼看到兄弟HAK180烫金机,如非提前知晓这是一台烫金机,可能会让人误以为是一台馈纸式扫描仪,毕竟从外观来看,兄弟HAK180烫金机与扫描仪有着相似的外观,尤其是进纸、出纸托盘的设 计,都有着一定相似度。 机身顶部的进纸托盘可以存放大量用于烫印的纸张,HAK180支持多种纸张质量规格,像办公常用70g/m²的A4纸张,以及更厚更重350g/m²的A4纸张都是可以正常实现烫印的,其中90g/m²纸张可以同时存放44张,350g/m²纸张可以同时存放12张,并可实现纸张自动、连续进纸烫金(如文章起始视频所示),这得益于其采用的“多页连续烫金”技术,可以处理批量烫印任务。 兄弟HAK180烫金机还支持“无版烫金”,整个烫印过程无需提前制版。如上图所示,比如我们需要制作一张用于表彰员工,或是学生的荣誉证书/奖 状,只需提前制作一张《荣誉证书》的样式(设计图),利用激光打印机,将样式内容打印出来,再将带有内容的一侧,面向HAK180放入到进纸托盘中,点击启动键后,HAK180可自动识 别激光打印机打印的内容,并在激光打印机打印的内容上,进行烫印工作,只需静待数秒,烫印成品就可从出纸托盘输出。",
                "url": "https://www.163.com/dy/article/HBO219SA05118VMB.html",
                "source": "web",
            },
            {
                "title": "兄弟(中国)发布烫金机,满足高端文印需求",
                "content": "在邀请函、贺卡或者是红包上轻松呈现出流光溢彩的烫金效果,随着兄弟(中国)23日正式推出的Brother HAK180烫金机而成为现实。这款体形轻巧烫 印机的面市,改变了以往需要到工厂定制才能实现烫金文印品的历史。个人用户只需在纸张介质上使用激光打印机打印好内容,再放入兄弟烫金机HAK180中,即可实现一键烫印,省去繁杂的软件编辑、电脑连接过程。 近年来,文印市场逐渐呈现精细化发展趋势,拥有核心技术的兄弟烫金机HAK180则恰好是可以满足高端文印需求的一款产品。为了让更多用 户体验这股“金色能量”,兄弟(中国)携全新烫金机Brother HAK180,以“引领鎏金岁月,创新成就JIN界”为发布会主题,于12月23日,带来了一场线上多平台直播,线下多地共享的双 线联动发布会,向中国用户全方位展示烫金之美。 兄弟(中国)商业有限公司董事长兼总裁尹炳新先生在当天的发布会上介绍,相关数据显示,中国是全球烫金文印市场规模最大的国家,占据全球60%的份额,其次是德国和日本。基于中国庞大的市场潜力,为满足高端文印市场对于个性化烫金需求,解决繁杂制版工序及成本高企等诸多困扰,兄弟集团决定在中国市场推出 “便捷使用”和“精品烫印”于一身的烫金机。 尹炳新先生介绍,以往实现烫金效果需要把产品送往工厂,交由大型专业设备进行处理。而兄弟(中国)推出的HAK180烫金机体积小巧,无需制版,避免环境污染,操作简便。据了解,这款产品可瞬间实现烫金效果,可广泛适用于各类场景,如精美邀请函,高档菜单与座位卡,激励学子的金色奖状等各类需要高品质,个性化定制的场景,满足学校,商务公司,高档宴会酒店等多用户需求。 据介绍,HAK180烫金机广泛支持各类纸张,胜任各式复杂情况,可高效便捷地为用户完成繁重任务。另外,HAK180烫金机在无版烫金与读秒烫金的基础之上,采用“金”“银”“红”三色烫金薄膜设计,让烫金效果达到纤毫毕现的水准。无论是纤细线条,亦或微小字体,都能精准呈现 。清晰的烫印效果,杜绝棱角、毛边、断线、模糊等恼人问题。同时烫印的内容耐得住长期保存,即便用手指刮抠也不会掉色或脱落,高品质烫金将为用户带来无可替代的体验。 整体上,做为凝聚着百年企业核心技术的Brother HAK180烫金机集合了兄弟集团始终坚持的高质量与高性价比的产品力,赋能其“无版烫印”、“多页连续烫金”、“纤毫毕现品质呈 现”多重创新技术,以提升用户烫金体验,为高端文印与商务交流提供更优质、更创新的解决方案。 顺应市场的需求,兄弟(中国)面向中国市场推出的Brother HAK180烫金机,以“ 无版烫",
                "url": "https://www.thepaper.cn/newsDetail_forward_15996505",
                "source": "web",
            },
            {
                "title": "HAK180",
                "content": "HAK180 烫金机 零售价 面议 最大15PPM烫金速度  可选7PPM烫金速度  无版烫印  配备最大44页标准ADF进纸器  支持省膜模式  10字符x2行LCD液 晶屏  HAK180烫金机,凭借其高速、高品质、以及出色的细节小字烫印效果,成为定制化专属机型。可烫印90g/m²~350g/m²的A4各类型纸张,支持各类广泛的应用领域。 高效、稳定的进纸结构 配备44页标准ADF进纸器,支持90g/m²~350g/m²的各类纸张(普通纸、薄纸、再生纸、厚纸等),进纸通道结构稳定可靠,支持连续烫印。 * 350g/m²支持12页自动进 纸 * 最大支持44页进纸容量(90g/m²)烫印面朝下 高速连续烫金 HAK180针对不同厚度、介质的纸张提供两种可选烫金速度。15ppm满足普通规格纸张的高效烫金需求,7ppm适合 稍厚纸张的烫金。 10字符×2行LCD液晶屏 10字符×2行LCD液晶屏,2个自定义按键,操作直观,方便快捷。 产品规格  一般参数  正常工作环境(温度): 10 ~ 32 摄氏度(50 ~ 90 华氏度) 正常工作环境(相对湿度): 20 % ~ 80 % 机器尺寸: W 384.2mm×D 330.2mm×H 356.2mm 重量(含包装箱): 16.9kg 电源: 220~240 V 消费电力(烫印中): 少于340W 消费电力(待机中): 少于7W 消费电力(关机): 少于0.04W LCD液晶屏尺寸: 48.0mm×10.9mm 节省烫金膜功能: 支持(在省膜模式中“跳过”和“中间”功能, 仅适用全幅烫金膜盒) 烫印 参数  最大烫印速度 (A4): 最高达15 ppm 可选烫印速度(A4): 7 ppm 视频 烫金机-HAK180-烫印速度调整-7PPM 烫金机-HAK180-安装耗材 烫金机-HAK180-更换耗材",
                "url": "https://www.brother.cn/hak/hak180",
                "source": "web",
            },
            {
                "title": "高速高品质 定制化专属,兄弟HAK180烫金机让你的文印店抢占先机",
                "content": "Brother兄弟(以下简称“兄弟”)推出的HAK180烫金机凭借其高速、高品质、以及出色的细节小字烫印效果,成为定制化专属机型,专业实力为邀请函、 贺卡、请柬等个性化定制需求提供了更多的便利,最终帮助用户实现产业升级、促进文印服务往高端化发展。同时,无版烫印、支持省膜模式,大幅降低运营成本,使用效率更高, 免去使用者的顾虑,为业务保驾护航。   紧凑体积,简约外观 外观方面,这款HAK180烫金机产品给人以沉稳扎实的感觉。产品颜色为黑色,磨砂的质感使得产品在使用时不易留下指纹,更具耐磨性。一体机整体观感棱角分明,但机身边角处均采用了圆润的设计,很大程度避免了用户在使用时发生不必要的磕碰。烫金机正面采用斜面设计,使得操作更加便捷舒适,摁键设置不用半蹲操作。并且外观还获得了2021年的日本GOOD DESIGN奖。       操作面板采用经济性和操作性适中的10字符*2行LCD液晶屏+按键的方式,操作直观,方便 快捷。对于打印店快速、效率的工作环境来说,简洁明了的直观显示非常友好。       在体积方面,兄弟HAK180体积大小为384.2mm*330.2mm*356.2mm,作为一台无版烫金机,这样的机身体积可以摆放在室内桌子上的任何位置。",
                "url": "https://www.163.com/dy/article/HC5ISR9H05119GO7.html",
                "source": "web",
            },
            {
                "title": "烫金机",
                "content": "烫金机 用途 个性化定制 红包烫印 奖状烫印 菜单烫印 信头烫印 名片烫印 请柬/邀请函烫印 行业 礼品店 图文行业 特殊行业 婚庆公司 特殊功 能 节省烫金膜 两档烫印速度可选 ADF自动进纸 连续烫印 A4及以下烫印 最大烫印速度 15ppm 烫金机 HAK180 烫金机 最大15PPM烫金速度  可选7PPM烫金速度  无版烫印  配备最大44页标准ADF进纸器  支持省膜模式",
                "url": "https://www.brother.cn/hak",
                "source": "web",
            },
            {
                "title": "注册免费试用Brother便携式移动办公设备",
                "content": "三步轻松制作标签 180度旋转打印功能 选择适合的标签模板并编辑内容 可导出标签打印记录 将标签模板传输至标签打印机 可设置管理键的密码  使用标签打印机独立打印标签 适应不同工作环境 可与电脑连接使用 与电脑连接 使用标签制作软件制作标签。 也可独立使用 预先传送标签模板后 仅使用标签打印机便可打 印标签。 两种打印方式可选 “连续模式” 连续模式能够完美胜任大批量的粘贴工作。 “剥离模式” 无需手动剥离标签底纸的剥离模式,便于需要频繁粘贴的工作。 ※标签剥离器为选配件。 6MB大内存容量 ① 最多可储存99个模板文件 ② 最多可调用64999行Excel数据(储存一个模板时) 适用各种卷式热敏标签纸 搭载传输/反射传感器, 可对应成本较低 的127mm的卷式热敏标签纸。 宽111mm 重量仅1.64kg 可摆放在店铺柜台等狭小的场所,也可随身携带。 标配的有线LAN接口 可在工作现场作为共享打印机使用 使用网络管理实用程序──BRAdmin, 可将公司总部创建的标签模板传送并储存至分店的标签打印机。 品种丰富的选配件可供选择 <table><tr><th></th><th>标签剥离器</th><th>无线网络接口</th><th>蓝牙接口</th><th>锂离子充电电池</th><th>充电电池底座</th><th>触摸屏显示器</th><th>串口转换器 (RJ25转DB9M)</th></tr><tr><td>产品名称</td><td>PA-LP-001</td><td>PA-WI-001</td><td>PA-BI-001</td><td>PA-BT-4000LI</td><td>PA-BB-001</td><td>PA-TDU-001</td><td>PA-SCA-001</td></tr></table>",
                "url": "https://www.brother.cn/project/pj_campaign/pc/Function.html",
                "source": "web",
            },
            {
                "title": "欢迎您访问兄弟(中国)商业有限公司",
                "content": "穿面线 将线穿过导线孔,将线绕过导线杆下侧。 将线沿顺时针方向缠绕张力旋钮(绕一次)。 安装梭芯 将绕满的梭芯放入旋梭中,如视频所示方式进行安装。 安装帽框支架 取下标准支架,将帽框支架穿过臂架,拧紧上面的2个螺丝,拧紧下面的2个螺丝。 更换机针 关闭机器电源,以视频所示方式卸下机针,并安装新机针。 软件更新 按下自动穿线按钮时,打开主电源,插入存有更新包的U盘,选择USB介质按键,等待机器下载完毕即可。。 加油机 开机后触摸绣花机加油图标。打开旋梭盖,然后卸下内旋梭。在旋梭上加一滴缝纫机油。 兄弟为你标记未来 服务协议 亲爱的用户: 感谢您使用Brother产品及服务! 为了持续向您提供更加丰富的产品功能,更智能、高效的使用体验,不断提升用户满意度,基于产品功能运行与服务的迭代需求, 我们需要对您在使用Brother产品及服务的过程中,产生的必要数据(如设备信息、配置参数等)进行收集、处理。 这些数据将有助于我们精准优化打印效率、提升响应速度及满足个性化需求,让每一次操作都更贴合您的期望。  我们深知,您的信任是我们最宝贵的财富!我们非常重视保护您的 个人信息、隐私及数据,始终将您的个人信息、隐私与数据安全置于首位, 为此我们将在严格遵守中华人民共和国《网络安全法》、《数据安全法》、《个人信息保护法》等法 律、行政法规的要求, 并在落实国家有关标准与行业实践的基础上,以合法、正当、必要、诚信为原则,依法保护、处理您的个人信息、隐私及数据。  为了更好地保障您的知情权、选择权等合法权益,帮助您清晰、完整地了解我们如何收集、保护、处理您的个人信息, 特别是涉及未成年人信息等重要场景,在您使用本产品及服务前,请务必全面、详细 阅读以下“产品使用与隐私保护”、 “数据保护与处理”、“未成年人信息保护”中的所有内容!这既是对您自身权益的积极保护,也是对我们工作的有力支持。",
                "url": "https://www.brother.cn/minisite/hsm/hsmschool/P&H/video.html",
                "source": "web",
            },
            {
                "title": "Brother商用绣花机专业成就非凡",
                "content": "了解整体产品的功能特点 行业应用 工艺/培训 产品特性 帮助您了解整体产品的功能特点,可以更好的进行选购 PR1055X 查看详情 操作视频 查看 配件 点击购买 10色针头,减少换线次数,一幅作品,不用换线, 10色线色随意搭配,提高效率。 自动穿线,大大提高了工作效率,省时省心,独有的摄像头定位,更准确在衣物面料 上找准刺绣位置,尤其是印花面料,绣花框感应系统,提升刺绣时的安全性,确保绣花不会超出既定范围 展开 PR680W 查看详情 操作视频 查看配件 点击购买 6色针头,减少换线 次数,一幅作品,不用换线, 6色线色随意搭配,提高效率。 自动穿线,大大提高了工作效率,省时省心,全新升级的十字激光定位灯,实现快速精准的定位,装夹布料更加便捷,绣花 框感应系统,提升刺绣时的安全性,确保绣花不会超出既定范围。 展开 VR 查看详情 操作视频 查看配件 点击购买 单个针头,绣名好帮手, 一体式梭芯绕线器,随时随地,想用就用。 自动穿线,大大提高了工作效率,省时省心, 绣花框感应系统,提升刺绣时的安全性, 确保绣花不会超出既定范围。 为您介绍行业使用产品的经典案例,快速了解产品应用场景 1 服装工厂 用于服装的定制(绣名)与打样 查看详情 2 专卖店 用于成衣、鞋帽定制 查看详情 3 毛绒玩具 用于毛绒玩具定制刺绣 查看详情 4 宠物周边 用于宠物周边场 景 查看详情 5 个人创业者 用于成品加工定制 查看详情 6 服装织补 用于绣补高档衣物 查看详情",
                "url": "https://www.brother.cn/minisite/PRpromotion/index.html",
                "source": "web",
            },
            {
                "title": "手动进纸",
                "content": "手动进纸 如果想从手动进纸槽进行打印,请转到情况 A: 从手动进纸槽打印. 如果不想从手动进纸槽进行打印,请转到情况 B: 从纸盒(纸盒1)打印. 情况A: 从手动进纸槽进行打印. 展开纸张支撑翼板以防止纸张从出纸托板中滑落,或者出纸后立即取走打印出的纸张。 请执行以下操作中的一项: 如果您的设备没有手动进纸 槽盖,请转到步骤3. 如果您的设备有手动进纸槽盖,请打开手动进纸槽盖。 用双手滑动手动进纸槽的纸张导块,调整至所用纸张的宽度。 用双手将一张纸放入手动进纸槽,直至 纸张的前缘触碰到进纸辊。 当感觉到设备进纸时请松开双手。将纸张以打印面向上的方式放入手动进纸槽。 设备将吸住纸张直至您发送打印数据到设备。 将打印数据发送至 设备前,请执行以下操作: 在标签上打印时: 打开后盖 (面朝上后出纸托板)。 在信封上打印时: 打开后盖 (面朝上后出纸托板),按下后盖内的两侧绿色锁定杆 。点击这里查看详情. 将打印出文档。如果仍出现错误信息,请转到步骤7. 放入手动进纸槽中的纸张尺寸可能与您在打印驱动中所选的纸张尺寸略有不同。请检查纸张尺寸。点击这里查看如何检查或更改纸张尺寸的详细信息. 如果想从手动进纸槽进纸并在标签或信封上打印,点击这里查看如何在标签和信封上打印的详细信息. 确保您所使用的纸张符合Brother推荐的纸张规格。点击这里查看推荐纸张的详细信息.点击这里查看您可使用的纸张类型. 装入与当前驱动设置相同尺寸的纸张。 将打印出文档。如果您想更改纸张来源,请转到情况 B中的步骤2. 情况B: 从纸盒(纸盒1)进行打印. 取消打印作业。 按下Go键4秒左右直到LED指示灯亮起,松开此键。 再次按下Go键。当取消打印作业时Ready和ErrorLED指示灯将闪烁。 确保手动进纸槽中未放置纸张。 若在手动进纸槽中放置纸张,即使在打印机驱动程序中选择了其他纸张来源,文档也将从手动进纸槽进行打印。 请执行以下操作中的一 项: 如果您仅想为下一次打印临时更改设置,请转到选项 1. 如果您想为所有打印作业更改默认纸张来源(纸盒),请转到选项 2. 选项1: 仅下一次打印临时更改设置 Windows 用户/Macintosh用户 Windows 用户: 注: 由于操作系统不同,操作步骤及屏幕显示可能也不同。 从您使用的应用程序选择打印菜单。 (使用的应用程序不同,有关选择打印菜单的步骤也不同.) 点击属性. 点击基本(Basic)选项卡并从纸张来源(Paper Source)下拉列表中选择纸盒1(Tray1). 点击确定(OK)将打印数据发送至设备。",
                "url": "https://support.brother.com/g/b/faqendbranchprintable.aspx?c=cn&lang=zh&prod=hl2250dn_eu_as&faqid=faq00002216_001&printable=true",
                "source": "web",
            },
            {
                "title": "FAQs & Troubleshooting",
                "content": "How do I wind the bobbin?  4 Stitches are skipped 5 How to adjust the thread tension  6 Combination of fabric, thread and needle 7 Tips for sewing an even seam allowance  8 Various sewing and application [Video instructions]  Fabric puckers 10 Touch panel is malfunctioning.  11 The needle contacts the needle plate.  12 How to make adjustments to character or decorative stitch patterns  13 Basic procedure to sew stitches 14 How to sew reverse stitches or reinforcement stitches  15 How to use the foot controller with the machine  16 Machine does not start to sew.  17 The stitch is not sewn correctly 18 How do I hoop the fabric in the embroidery frame? 19 How do I sew hook-and-loop fasteners(hook-and-loop tapes)? 20 Upper thread breaks.  21 Bobbin thread breaks  22 23 How do I remove or attach the Embroidery foot ?  24 How do I set the bobbin?  25 26   2 8",
                "url": "https://support.brother.com/g/b/faqlist.aspx?c=nz&lang=en&prod=hf_inovnv180eas&tabid=2",
                "source": "web",
            },
        ],
    }
    res = node(init_state)

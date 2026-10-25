# atguigu/query_process/nodes/node_rrf.py

from atguigu.tool.json_format_tool import json_format
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
from observer.collector import trace_collector
import time


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（Query、HyDE）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState):
        """
        对多路召回(重写Query和HyDE)的结果进行加权融合排序。
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        task_id = state.get("task_id") or ""
        t_start = time.perf_counter()

        hyde_embedding_chunks = state.get("hyde_embedding_chunks", "")
        embedding_chunks = state.get("embedding_chunks", "")
        dense_chunks = state.get("dense_chunks") or []
        sparse_chunks = state.get("sparse_chunks") or []
        if not hyde_embedding_chunks or not embedding_chunks:
            logger.error(f"没有召回结果: {embedding_chunks=},{hyde_embedding_chunks=}")
            raise ValueError(f"没有召回结果: {embedding_chunks=},{hyde_embedding_chunks=}")

        RRF_SMOOTH = 60
        RRF_TOP_K = 10
        rrf_dict = {}  # 用 dict 代替列表，实现 O(1) 复杂度的去重与定位
        # 对 Re-Query 混合检索与 HyDE 混合检索进行 1:1 倒排融合，同时将未进混合前10的 Dense/Sparse 尾部候补纳入展示
        rrf_weight = [(embedding_chunks, 1.0), (hyde_embedding_chunks, 1.0)]
        if dense_chunks:
            rrf_weight.append((dense_chunks, 0.3))
        if sparse_chunks:
            rrf_weight.append((sparse_chunks, 0.3))

        for embedding, weight in rrf_weight:
            for rank, chunk in enumerate(embedding, 1):
                chunk_id = chunk.get("id")
                if chunk_id is None:
                    # 容错处理：若 chunk 缺失 id，记录警告或跳过，防止程序崩溃
                    logger.warning(f"Chunk 缺少 'id' 字段，跳过该项: {chunk}")
                    continue

                score_rrf = weight / (rank + RRF_SMOOTH)

                if chunk_id not in rrf_dict:
                    # 通过 copy() 创建新字典，避免污染原始数据
                    new_chunk = chunk.copy()
                    new_chunk["score_rrf"] = score_rrf
                    rrf_dict[chunk_id] = new_chunk
                else:
                    rrf_dict[chunk_id]["score_rrf"] += score_rrf

        # 按得分排序，并转回列表结构
        rrf_all_chunks = sorted(rrf_dict.values(), key=lambda x: x["score_rrf"], reverse=True)
        for rank_idx, item in enumerate(rrf_all_chunks, 1):
            item["rrf_rank"] = rank_idx

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        if task_id:
            trace_collector.add_io_item(task_id, "rrf", "RRF 倒排融合 CPU 计算", elapsed_ms, io_type="cpu")
            trace_collector.set_module_total_ms(task_id, "rrf", elapsed_ms)
            trace_collector.record_rrf(task_id, rrf_all_chunks, top_k=RRF_TOP_K, rrf_smooth=RRF_SMOOTH)

        # 截取 Top-K 传递给下游重排（若未超 Top-K 则全量传递）
        rrf_chunks = rrf_all_chunks[:RRF_TOP_K]

        return {"rrf_chunks": rrf_chunks}


if __name__ == "__main__":
    mock_state = {
        "embedding_chunks": [
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![设备需放置于平稳通风处，避免震动；搬运时双手托底，勿触危险区域；使用后断电，注意纸张边缘锋利。](http://192.168.100.88:9000/knowledge-base/upload-images/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788527,
                "score": 0.830259382724762,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![本设备需接地使用，放置于平稳通风处，避免灰尘堆积和手指误入危险区域，搬运时用双手抓稳。](http://192.168.100.88:9000/knowledge-base/upload-images/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788526,
                "score": 0.8287613987922668,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![设备使用需注意防火、防触电，避免儿童接触塑料袋，使用后待冷却再开盖，防止烧伤。](http://192.168.100.88:9000/knowledge-base/upload-images/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788521,
                "score": 0.8251358270645142,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788513,
                "score": 0.8236921429634094,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![确保设备放置平稳，远离边缘，使用时勿将手伸入纸张边缘，搬运需双手托底，避免跌落造成伤害或损坏。](http://192.168.100.88:9000/knowledge-base/upload-images/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788530,
                "score": 0.8157209753990173,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788514,
                "score": 0.8140485882759094,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "无标题",
                "content": "![HAK 180烫金机产品安全手册，含使用前须知、安全提示及获取说明书的官方网址。](http://192.168.100.88:9000/knowledge-base/upload-images/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN\n",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788512,
                "score": 0.8104063868522644,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788517,
                "score": 0.8088258504867554,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![使用起搏器者需远离设备，注意高温部件防烫伤；设备须接220-240V交流电，禁用直流电源，防止触电或火灾。](http://192.168.100.88:9000/knowledge-base/upload-images/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788522,
                "score": 0.8076007962226868,
                "source": "local",
            },
            {
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。",
                "item_name": "BrotherHAK180烫金机",
                "id": 468273558621788516,
                "score": 0.6857563853263855,
                "source": "local",
            },
        ],
        "hyde_embedding_chunks": [
            {
                "id": 468273558621788521,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![设备使用需注意防火、防触电，避免儿童接触塑料袋，使用后待冷却再开盖，防止烧伤。](http://192.168.100.88:9000/knowledge-base/upload-images/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8514308929443359,
                "source": "local",
            },
            {
                "id": 468273558621788527,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![设备需放置于平稳通风处，避免震动；搬运时双手托底，勿触危险区域；使用后断电，注意纸张边缘锋利。](http://192.168.100.88:9000/knowledge-base/upload-images/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8473448753356934,
                "source": "local",
            },
            {
                "id": 468273558621788513,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8460150957107544,
                "source": "local",
            },
            {
                "id": 468273558621788514,
                "file_title": "hak180产品安全手册",
                "title": "## HAK 180 烫金机",
                "content": "## HAK 180 烫金机\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8388824462890625,
                "source": "local",
            },
            {
                "id": 468273558621788522,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n![使用起搏器者需远离设备，注意高温部件防烫伤；设备须接220-240V交流电，禁用直流电源，防止触电或火灾。](http://192.168.100.88:9000/knowledge-base/upload-images/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8288931846618652,
                "source": "local",
            },
            {
                "id": 468273558621788530,
                "file_title": "hak180产品安全手册",
                "title": "## 为设备选择一个安全的位置",
                "content": "## 为设备选择一个安全的位置\n\n![确保设备放置平稳，远离边缘，使用时勿将手伸入纸张边缘，搬运需双手托底，避免跌落造成伤害或损坏。](http://192.168.100.88:9000/knowledge-base/upload-images/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8283040523529053,
                "source": "local",
            },
            {
                "id": 468273558621788532,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.819697380065918,
                "source": "local",
            },
            {
                "id": 468273558621788517,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.8154225945472717,
                "source": "local",
            },
            {
                "id": 468273558621788512,
                "file_title": "hak180产品安全手册",
                "title": "无标题",
                "content": "![HAK 180烫金机产品安全手册，含使用前须知、安全提示及获取说明书的官方网址。](http://192.168.100.88:9000/knowledge-base/upload-images/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN\n",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.7273162007331848,
                "source": "local",
            },
            {
                "id": 999999999999999999,
                "file_title": "hak180产品安全手册",
                "title": "## 设备",
                "content": "## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![本设备需接地使用，放置于平稳通风处，避免灰尘堆积和手指误入危险区域，搬运时用双手抓稳。](http://192.168.100.88:9000/knowledge-base/upload-images/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "score": 0.6666666666666,
                "source": "local",
            },
        ],
    }
    node = NodeRrf()
    res = node(mock_state)
    print(json_format(res["rrf_chunks"]))
    print(len(res["rrf_chunks"]))

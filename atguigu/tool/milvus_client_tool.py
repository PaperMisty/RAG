from pymilvus import RRFRanker
from pymilvus import WeightedRanker
from pymilvus import MilvusClient, AnnSearchRequest
from atguigu.config.config import MilvusConfig
from atguigu.tool.logger import logger


client = None


def get_milvus_client() -> MilvusClient:
    """获取MilvusClient实例"""
    global client
    if not client:
        try:
            client = MilvusClient(uri=MilvusConfig.milvus_url)
            logger.info("Milvus连接成功")
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise e
    return client


# 创建混合检索的两个 ANN Search Request
def create_reqs(
    dense_data,
    sparse_data,
    dense_ann_field=None,
    sparse_ann_field=None,
    dense_params=None,
    sparse_params=None,
    limit=10,
    expr=None,
) -> list:
    """
    创建混合检索的ANN Search Request

    Args:
        dense_data (list): 稠密向量数据
        sparse_data (list): 稀疏向量数据
        dense_ann_field (str, optional): 稠密向量搜索字段名. Defaults to None.
        sparse_ann_field (str, optional): 稀疏向量搜索字段名. Defaults to None.
        dense_params (dict, optional): 稠密向量参数. Defaults to None.
        sparse_params (dict, optional): 稀疏向量参数. Defaults to None.
        limit (int, optional): 返回结果数量. Defaults to 10.
        expr (str, optional): 标量搜索过滤表达式. Defaults to None.

    Returns:
        list: 混合检索请求
    """
    if dense_params is None:
        dense_params = {
            "metric_type": "COSINE"
        }  # 检索用的度量方式需要和milvus集合建立时的向量类型一致,否则报错,这里默认取余弦相似度
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}
    dense_ann_req = AnnSearchRequest(
        data=[dense_data], anns_field=dense_ann_field, param=dense_params, limit=limit, expr=expr
    )
    sparse_ann_req = AnnSearchRequest(
        data=[sparse_data], anns_field=sparse_ann_field, param=sparse_params, limit=limit, expr=expr
    )
    return [dense_ann_req, sparse_ann_req]


# 混合检索搜索
def weighted_hybrid_search(
    collection_name,
    reqs,
    ranker=[0.5, 0.5],
    limit=10,
    output_fields: list = None,
) -> list:
    """混合检索搜索, 采用分数归一化并加权方式进行融合排序

    Args:
        collection_name (str): 集合名称
        reqs (list): 混合检索请求
        ranker (list, optional): 混合检索权重. Defaults to [0.5, 0.5].
        limit (int, optional): 返回结果数量. Defaults to 10.
        output_fields (list, optional): 返回结果字段. Defaults to None.

    Returns:
        list: 混合检索结果
    """
    ranker = WeightedRanker(*ranker, norm_score=True)  # 这个类默认不会进行归一化?而是我们必须设计表的时候采用了归一化
    # ranker = RRFRanker()  # 采用倒数排名排序
    client = get_milvus_client()
    # 显式加载集合到内存中，防止检索结果因未载入而为空
    client.load_collection(collection_name)
    result = client.hybrid_search(
        collection_name=collection_name, reqs=reqs, ranker=ranker, limit=limit, output_fields=output_fields
    )
    return result


def search_dense_and_sparse_separately(
    collection_name: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    expr: str = "",
    limit: int = 10,
    output_fields: list[str] | None = None,
    source: str = "local",
) -> tuple[list[dict], list[dict], float, float]:
    """职责: 复用Milvus连接池分别执行Dense稠密检索与Sparse稀疏检索并返回各自排名切片与IO耗时(ms)"""
    import time

    milvus_cli = get_milvus_client()
    milvus_cli.load_collection(collection_name)

    # 1. 执行 Dense 单路余弦相似度检索并记录网络IO耗时
    t_dense = time.perf_counter()
    dense_res = milvus_cli.search(
        collection_name=collection_name,
        data=[dense_vec],
        anns_field="dense",
        filter=expr or "",
        limit=limit,
        output_fields=output_fields,
        search_params={"metric_type": "COSINE"},
    )
    dense_io_ms = round((time.perf_counter() - t_dense) * 1000, 2)
    dense_chunks: list[dict] = []
    if dense_res and len(dense_res) > 0:
        for hit in dense_res[0]:
            dense_chunks.append({**hit.get("entity", {}), "score": hit.get("distance", 0.0), "source": source})

    # 2. 执行 Sparse 单路内积检索并记录网络IO耗时
    t_sparse = time.perf_counter()
    sparse_res = milvus_cli.search(
        collection_name=collection_name,
        data=[sparse_vec],
        anns_field="sparse",
        filter=expr or "",
        limit=limit,
        output_fields=output_fields,
        search_params={"metric_type": "IP"},
    )
    sparse_io_ms = round((time.perf_counter() - t_sparse) * 1000, 2)
    sparse_chunks: list[dict] = []
    if sparse_res and len(sparse_res) > 0:
        for hit in sparse_res[0]:
            sparse_chunks.append({**hit.get("entity", {}), "score": hit.get("distance", 0.0), "source": source})

    return dense_chunks, sparse_chunks, dense_io_ms, sparse_io_ms


if __name__ == "__main__":
    get_milvus_client()

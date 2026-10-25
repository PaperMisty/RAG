# 基于 LangGraph 的模块化高级 RAG 系统 (Advanced & Modular RAG System)

本项目是一个基于 Python 和有向无环图（DAG，通过 **LangGraph** 状态机编排）的**模块化/高级 RAG（Advanced & Modular RAG）生产级架构原型**。

项目将数据流水线（数据提取、非结构化解析、多路向量入库）与检索生成流水线（意图确认、多路召回、重排、大模型生成）进行了高内聚、低耦合的 `Node` 级别解耦，旨在解决传统 RAG 系统在面对 PDF 图表提取困难、多路召回相关性差以及多轮对话历史状态丢失等核心痛点。

---

## 🛠️ 技术栈

* **图编排与状态机**：`LangGraph` (实现基于图的多节点协作与条件路由)
* **向量数据库**：`Milvus` (实现稠密向量 + 稀疏向量的混合索引入库与双路召回)
* **文档高保真提取**：`Mineru API` (实现 PDF 高保真提取为结构化 Markdown)
* **图片多模态理解**：`VLM` (视觉大语言模型，实现图片上下文图表摘要提取)
* **对象存储**：`MinIO` (托管 RAG 文档内部的多媒体/图片，实现外链动态回填)
* **历史对话与状态持久化**：`MongoDB` (带有 `[("session_id", 1), ("ts", -1)]` 复合索引的高性能历史会话存储)
* **嵌入与重排模型**：`BGE-M3` (双路 Embedding 生成) + `Cross-Encoder` (精确重排 Rerank)

---

## 📐 架构设计与节点流转

项目在 `nodes/` 目录下将复杂的 RAG 系统高度解耦，划分为两大核心工作流图：

### 1. 数据导入流水线 (Import Graph Pipeline)
共包含 **7 ~ 9 个节点**，实现非结构化 PDF 文档的完美解析与多模态向量化入库：
* **`NodeEntry`**：入口节点，智能判断输入文件格式（PDF / MD）并执行条件分流。
* **`NodePDFToMD`**：调用 Mineru 将 PDF 提取为高保真 Markdown。**内置代理屏蔽降级机制，当 requests 发生 SSL 劫持报错时，自动无感切换至系统 `curl` 进程拉取 ZIP 文件**。
* **`NodeMDImg`**：图片理解节点。提取 Markdown 中的所有图片，结合上下文通过 VLM生成 50 字以内的图片摘要；将图片上传至 MinIO 对象存储，并将新 Markdown（`*_new.md`）中的图片外链替换为 MinIO 线上 URL。
* **`NodeDocumentSplit`**：文档切分节点。采用父子文档递归式切分，最大字符长度设为 65535，防止长文本溢出。
* **`NodeItemNameRecognition`**：物料/主体名称识别节点。调用大模型自动提取该文档对应的主体类别（例如万用表、烫金机等）。
* **`NodeBGEEmbedding`**：嵌入向量生成节点。利用 BGE-M3 生成稠密（Dense）与稀疏（Sparse）混合双路向量。
* **`NodeImportMilvus`**：Milvus 向量导入节点。**执行幂等性删除（以文件名作为过滤标准进行前置清理），并重新将双路向量与元数据插入 Milvus 中**。

### 2. 检索生成流水线 (Query Graph Pipeline)
共包含 **7 ~ 9 个节点**，实现精准的意图对齐、多路检索与分流：
* **`NodeItemNameConfirm`**：意图识别与主体确认节点。提取商品主体并与 Milvus 中商品库执行层次对齐，输出置信度指标。
* **分流路由决策 (`after_item_name_confirm_router`)**：
  * **高置信度**：成功锁定本地主体，分发至本地双路混合检索分支。
  * **低置信度**：未匹配到本地主体，走 `NodeWebSearchMcp` 联网检索与重排。
  * **中置信度**：匹配到多个歧义候选，生成澄清追问直接返回给用户。
* **`NodeSearchEmbedding`**：Re-Query 混合检索节点。调用 BGE-M3 生成 Dense 与 Sparse 向量，在 Milvus 中执行标量过滤搜索，并采用**跨度量 Min-Max 归一化**算法进行加权融合。
* **`NodeSearchEmbeddingHyde`**：HyDE 假设性检索节点。大模型先生成假设性解答，再执行 BGE-M3 双路向量化与 Milvus 检索。
* **`NodeRrf`**：倒排排名融合节点。加权融合 Re-Query 混合检索与 HyDE 检索结果，并保留 Dense/Sparse 候补切片。
* **`NodeRerank`**：重排与断崖截断节点。拼接文档标题与章节标题，使用 Cross-Encoder 精排模型打分，并根据得分断崖自动过滤低相关性切片。
* **`NodeAnswerOutput`**：应答生成节点。组装携带文档与章节上下文的切片，调用大模型生成答案，并将多轮会话持久化到 MongoDB。

---

## ⚡ 核心优势与特色设计

1. **多模态图表无损保留**：传统的 RAG 抛弃了 PDF 图表，本项目通过 VLM 生成图表文字摘要，并用对象存储替换外链，使图表信息在向量库里“可检索、可理解”。
2. **跨度量 Min-Max 归一化混合检索**：稠密向量（Cosine/IP）与稀疏向量（BM25/IP）得分量纲差异极大。系统采用 Min-Max 动态归一化算法统一量纲后加权融合，充分发挥语义与关键词双重召回能力。
3. **Re-Query + HyDE 双路加权 RRF 融合与重排断崖截断**：
   * 将 Re-Query 混合检索与 HyDE 假设检索进行 1:1 倒排融合。
   * 采用 Cross-Encoder 重排，并引入断崖检测机制，自动剔除得分急剧衰减的噪音切片。
4. **切片上下文元数据自动增强**：在重排及最终 Prompt 构建阶段，自动在切片前附加 `文档标题 | 章节标题`，解决纯表格与段落切片上下文缺失的缺陷。
5. **全链路实时观测台 (Observer Dashboard & Tracing)**：
   * 内置 `TraceCollector` 追踪收集器，精准采集各节点耗时与细分 IO 耗时（网络 HTTP、数据库 Milvus/MongoDB、LLM 推理、CPU 计算）。
   * 前端集成可视化观测画布，实时展示路由决策分支、召回候选项详情与首字时间（TTFT）。
6. **极强健壮性的 MongoDB 写入机制**：在 `add_or_update_data` 方法中，自动对 PyMongo 的 `Database` 对象做布尔测试修正。通过 if-else 分流实现高业务安全的增改语义隔离：当传错 `_id` 找不到数据时**绝不静默新增**产生脏数据；且自动从更新内容中剥离 `_id` 主键并使用 `{"$set": ...}` 进行安全的局部更新。
7. **全自动防网络劫持下载 fallback**：在连接国内 CDN 频繁由于代理 TUN 接管触发 SSL 握手 EOF 重置时，自动切换至系统的 `curl` 原生进程强制拉取，极大地提高了本地复杂代理开发环境下的鲁棒性。

---

## 🗺️ 架构流程图 (Mermaid)

项目根目录提供了严格遵循规范的架构流程图文件：[query_arch.mmd](file:///z:/home/ubuntu/projects/RAG/kb_0515/query_arch.mmd)，清晰展现了输入意图识别、路由决策、高/低置信度链路、RRF 融合及重排流转。

---

## 🔮 它距离“超大规模企业级系统”还有多远？

虽然在**算法思路和节点解耦**上本项目已达到生产级，但要在并发量极高的超大型企业环境中落地，它还需要补充以下三个工程硬实力：

### 1. 高并发的异步流式处理 (Concurrency & Streaming)
当前的节点逻辑中包含大量的阻塞式网络请求（如同步的 `requests`，同步的 `MongoClient` 等）。在高并发场景下会导致明显的线程池枯竭。在线上部署时需将整个 LangGraph 的节点函数改造为全异步模式（`async/await`），使用 `AsyncMongoClient` 替代同步客户端，并在最后的生成节点支持流式（Streaming）字符输出。

### 2. 高频写入缓冲 (Message Queue)
在海量 PDF 大批量入库或者高频实时会话写入时，为了防止数据库（MongoDB、Milvus、MinIO）瞬间被流量洪峰压垮，需要引入 `Kafka` 或 `RabbitMQ` 这样的消息队列来进行削峰填谷与异步解耦。

### 3. 数据评估体系与大模型监控 (LLM Ops / Evaluation)
真正的工业化大模型系统会配备持续的评估大盘（如 `Ragas`、`TruLens`）。通过在线收集真实用户的 Question 与 LLM Answer，定期计算检索的召回率（Recall）、上下文相关度（Context Precision）以及回答忠实度（Faithfulness），根据评估指标来反向微调各个节点的超参数（例如 Chunk 大小、检索 `Top-K` 阀值等）。

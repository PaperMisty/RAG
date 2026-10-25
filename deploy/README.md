# RAG 基础设施服务运维手册 (WSL2 / Local Production)

本目录承载当前 RAG 系统的核心外部服务栈（Milvus 向量库、MinIO 对象存储、etcd 元数据中心、MongoDB 会话库及 Attu 可视化面板）。

---

## 目录结构规划

```text
deploy/
├── docker-compose.yml     # 容器服务编排定义
├── README.md              # 运维操作说明
└── volumes/               # 持久化数据目录（已加入 .gitignore，禁止提交仓库）
    ├── etcd/              # etcd 元数据
    ├── minio/             # MinIO 图片与向量底层数据
    ├── milvus/            # Milvus 引擎运行时数据
    └── mongodb/           # MongoDB 会话与聊天历史数据
```

---

## 服务端口与访问入口

| 服务名称 | 容器内服务 | 宿主映射端口 | 访问方式 / 协议 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Milvus** | 向量数据库 | `19530` | gRPC / SDK (`http://127.0.0.1:19530`) | 用于 Python 客户端连接 |
| **Attu** | Milvus 可视化面板 | `8000` | Web (`http://127.0.0.1:8000`) | 浏览器登录可直观查看集合与切片 |
| **MinIO API** | 对象存储 API | `9000` | HTTP (`http://127.0.0.1:9000`) | 用于图片上传与读取 |
| **MinIO Console**| 对象存储管理后台 | `9001` | Web (`http://127.0.0.1:9001`) | 账号/密码: `minioadmin` / `minioadmin` |
| **MongoDB** | 会话上下文存储 | `27017` | MongoDB 协议 (`127.0.0.1:27017`) | 存储多轮会话上下文 |

---

## 常用运维命令

### 1. 一键后台启动所有服务
在项目根目录下或进入 `deploy/` 目录执行：
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### 2. 检查服务健康状态
```bash
docker compose -f deploy/docker-compose.yml ps
```
> 所有容器状态应显示为 `Up` 且标注 `(healthy)`。

### 3. 查看实时日志
```bash
# 查看所有服务实时日志
docker compose -f deploy/docker-compose.yml logs -f

# 仅查看 Milvus 日志
docker compose -f deploy/docker-compose.yml logs -f standalone
```

### 4. 停止所有服务（数据保留）
```bash
docker compose -f deploy/docker-compose.yml stop
```

### 5. 彻底重置并清理全部数据（慎用）
```bash
docker compose -f deploy/docker-compose.yml down
sudo rm -rf deploy/volumes/*
```

---

## 项目环境对接（.env 调整）

服务在 WSL2 启动后，修改项目根目录的 `.env`：

```ini
# MinIO
MINIO_ENDPOINT = 127.0.0.1:9000
MINIO_ACCESS = minioadmin
MINIO_SECRET = minioadmin
MINIO_BUCKET_NAME = knowledge-base
MINIO_IMG_DIR = upload_images

# Milvus
MILVUS_URL = http://127.0.0.1:19530
CHUNKS_COLLECTION = kb_chunks
ITEM_NAME_COLLECTION = kb_item_names

# MongoDB
mongo_url = 127.0.0.1:27017
mongo_db_name = kb_message
```

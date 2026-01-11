# Serverless Vectorizer

[![CI](https://github.com/johnnywale/serverless-vectorizer/actions/workflows/test.yml/badge.svg)](https://github.com/johnnywale/serverless-vectorizer/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/johnnywale/serverless-vectorizer)](https://github.com/johnnywale/serverless-vectorizer/releases)
[![License](https://img.shields.io/github/license/johnnywale/serverless-vectorizer)](LICENSE-MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/johnnywale/serverless-vectorizer)](https://hub.docker.com/r/johnnywale/serverless-vectorizer)
[![Docker Image Size](https://img.shields.io/docker/image-size/johnnywale/serverless-vectorizer/latest)](https://hub.docker.com/r/johnnywale/serverless-vectorizer)

AWS Lambda container image for generating embeddings using [fastembed-rs](https://github.com/Anush008/fastembed-rs). Supports **text embeddings**, **image embeddings**, **sparse embeddings**, and **reranking models**. Models are pre-loaded into Docker images for fast cold starts.

## Prebuilt Docker Images

The following text embedding models have prebuilt Docker images available on Docker Hub. You can pull and use them directly:

| Model                                        | Model ID                                              | Dimension | Description                                                                | Docker Image                                                                                   |
|----------------------------------------------|-------------------------------------------------------|-----------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| All-MINILM-L12-v2                            | `Xenova/all-MiniLM-L12-v2`                            | 384       | `Quantized Sentence Transformer model, MiniLM-L12-v2`                      | `johnnywalee/serverless-vectorizer:latest-Xenova/all-MiniLM-L12-v2`                            |
| Snowflake-Arctic-Embed-Xs                    | `snowflake/snowflake-arctic-embed-xs`                 | 384       | `Snowflake Arctic embed model, xs`                                         | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-xs`                 |
| BGE-Small-ZH-v1.5                            | `Xenova/bge-small-zh-v1.5`                            | 512       | `v1.5 release of the small Chinese model`                                  | `johnnywalee/serverless-vectorizer:latest-Xenova/bge-small-zh-v1.5`                            |
| BGE-Small-EN-v1.5-Onnx-Q                     | `Qdrant/bge-small-en-v1.5-onnx-Q`                     | 384       | `Quantized v1.5 release of the fast and default English model`             | `johnnywalee/serverless-vectorizer:latest-Qdrant/bge-small-en-v1.5-onnx-Q`                     |
| Snowflake-Arctic-Embed-S                     | `snowflake/snowflake-arctic-embed-s`                  | 384       | `Quantized Snowflake Arctic embed model, small`                            | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-s`                  |
| Snowflake-Arctic-Embed-M-Long                | `snowflake/snowflake-arctic-embed-m-long`             | 768       | `Snowflake Arctic embed model, medium with 2048 context`                   | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-m-long`             |
| BGE-Base-EN-v1.5                             | `Xenova/bge-base-en-v1.5`                             | 768       | `v1.5 release of the base English model`                                   | `johnnywalee/serverless-vectorizer:latest-Xenova/bge-base-en-v1.5`                             |
| Snowflake-Arctic-Embed-M-Long                | `snowflake/snowflake-arctic-embed-m-long`             | 768       | `Quantized Snowflake Arctic embed model, medium with 2048 context`         | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-m-long`             |
| Paraphrase-Multilingual-MPNET-Base-v2        | `Xenova/paraphrase-multilingual-mpnet-base-v2`        | 768       | `Sentence-transformers model for tasks like clustering or semantic search` | `johnnywalee/serverless-vectorizer:latest-Xenova/paraphrase-multilingual-mpnet-base-v2`        |
| BGE-Large-ZH-v1.5                            | `Xenova/bge-large-zh-v1.5`                            | 1024      | `v1.5 release of the large Chinese model`                                  | `johnnywalee/serverless-vectorizer:latest-Xenova/bge-large-zh-v1.5`                            |
| Modernbert-Embed-Large                       | `lightonai/modernbert-embed-large`                    | 1024      | `Large model of ModernBert Text Embeddings`                                | `johnnywalee/serverless-vectorizer:latest-lightonai/modernbert-embed-large`                    |
| Multilingual-E5-Large-Onnx                   | `Qdrant/multilingual-e5-large-onnx`                   | 1024      | `Large model of multilingual E5 Text Embeddings`                           | `johnnywalee/serverless-vectorizer:latest-Qdrant/multilingual-e5-large-onnx`                   |
| BGE-Large-EN-v1.5                            | `Xenova/bge-large-en-v1.5`                            | 1024      | `v1.5 release of the large English model`                                  | `johnnywalee/serverless-vectorizer:latest-Xenova/bge-large-en-v1.5`                            |
| Multilingual-E5-Small                        | `intfloat/multilingual-e5-small`                      | 384       | `Small model of multilingual E5 Text Embeddings`                           | `johnnywalee/serverless-vectorizer:latest-intfloat/multilingual-e5-small`                      |
| Snowflake-Arctic-Embed-M                     | `Snowflake/snowflake-arctic-embed-m`                  | 768       | `Snowflake Arctic embed model, medium`                                     | `johnnywalee/serverless-vectorizer:latest-Snowflake/snowflake-arctic-embed-m`                  |
| GTE-Large-EN-v1.5                            | `Alibaba-NLP/gte-large-en-v1.5`                       | 1024      | `Large multilingual embedding model from Alibaba`                          | `johnnywalee/serverless-vectorizer:latest-Alibaba-NLP/gte-large-en-v1.5`                       |
| All-MPNET-Base-v2                            | `Xenova/all-mpnet-base-v2`                            | 768       | `Sentence Transformer model, mpnet-base-v2`                                | `johnnywalee/serverless-vectorizer:latest-Xenova/all-mpnet-base-v2`                            |
| Nomic-Embed-Text-v1                          | `nomic-ai/nomic-embed-text-v1`                        | 768       | `8192 context length english model`                                        | `johnnywalee/serverless-vectorizer:latest-nomic-ai/nomic-embed-text-v1`                        |
| All-MINILM-L6-v2                             | `Xenova/all-MiniLM-L6-v2`                             | 384       | `Quantized Sentence Transformer model, MiniLM-L6-v2`                       | `johnnywalee/serverless-vectorizer:latest-Xenova/all-MiniLM-L6-v2`                             |
| GTE-Base-EN-v1.5                             | `Alibaba-NLP/gte-base-en-v1.5`                        | 768       | `Large multilingual embedding model from Alibaba`                          | `johnnywalee/serverless-vectorizer:latest-Alibaba-NLP/gte-base-en-v1.5`                        |
| GTE-Large-EN-v1.5                            | `Alibaba-NLP/gte-large-en-v1.5`                       | 1024      | `Quantized Large multilingual embedding model from Alibaba`                | `johnnywalee/serverless-vectorizer:latest-Alibaba-NLP/gte-large-en-v1.5`                       |
| Clip-ViT-B-32-Text                           | `Qdrant/clip-ViT-B-32-text`                           | 512       | `CLIP text encoder based on ViT-B/32`                                      | `johnnywalee/serverless-vectorizer:latest-Qdrant/clip-ViT-B-32-text`                           |
| BGE-Base-EN-v1.5-Onnx-Q                      | `Qdrant/bge-base-en-v1.5-onnx-Q`                      | 768       | `Quantized v1.5 release of the large English model`                        | `johnnywalee/serverless-vectorizer:latest-Qdrant/bge-base-en-v1.5-onnx-Q`                      |
| BGE-Small-EN-v1.5                            | `Xenova/bge-small-en-v1.5`                            | 384       | `v1.5 release of the fast and default English model`                       | `johnnywalee/serverless-vectorizer:latest-Xenova/bge-small-en-v1.5`                            |
| Snowflake-Arctic-Embed-S                     | `snowflake/snowflake-arctic-embed-s`                  | 384       | `Snowflake Arctic embed model, small`                                      | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-s`                  |
| JINA-Embeddings-v2-Base-Code                 | `jinaai/jina-embeddings-v2-base-code`                 | 768       | `Jina embeddings v2 base code`                                             | `johnnywalee/serverless-vectorizer:latest-jinaai/jina-embeddings-v2-base-code`                 |
| Snowflake-Arctic-Embed-L                     | `snowflake/snowflake-arctic-embed-l`                  | 1024      | `Quantized Snowflake Arctic embed model, large`                            | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-l`                  |
| All-MINILM-L6-v2-Onnx                        | `Qdrant/all-MiniLM-L6-v2-onnx`                        | 384       | `Sentence Transformer model, MiniLM-L6-v2`                                 | `johnnywalee/serverless-vectorizer:latest-Qdrant/all-MiniLM-L6-v2-onnx`                        |
| Multilingual-E5-Base                         | `intfloat/multilingual-e5-base`                       | 768       | `Base model of multilingual E5 Text Embeddings`                            | `johnnywalee/serverless-vectorizer:latest-intfloat/multilingual-e5-base`                       |
| Paraphrase-Multilingual-MINILM-L12-v2-Onnx-Q | `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` | 384       | `Quantized Multi-lingual model`                                            | `johnnywalee/serverless-vectorizer:latest-Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` |
| GTE-Base-EN-v1.5                             | `Alibaba-NLP/gte-base-en-v1.5`                        | 768       | `Quantized Large multilingual embedding model from Alibaba`                | `johnnywalee/serverless-vectorizer:latest-Alibaba-NLP/gte-base-en-v1.5`                        |
| All-MINILM-L12-v2                            | `Xenova/all-MiniLM-L12-v2`                            | 384       | `Sentence Transformer model, MiniLM-L12-v2`                                | `johnnywalee/serverless-vectorizer:latest-Xenova/all-MiniLM-L12-v2`                            |
| BGE-M3                                       | `BAAI/bge-m3`                                         | 1024      | `Multilingual M3 model with 8192 context length, supports 100+ languages`  | `johnnywalee/serverless-vectorizer:latest-BAAI/bge-m3`                                         |
| Nomic-Embed-Text-v1.5                        | `nomic-ai/nomic-embed-text-v1.5`                      | 768       | `v1.5 release of the 8192 context length english model`                    | `johnnywalee/serverless-vectorizer:latest-nomic-ai/nomic-embed-text-v1.5`                      |
| Nomic-Embed-Text-v1.5                        | `nomic-ai/nomic-embed-text-v1.5`                      | 768       | `Quantized v1.5 release of the 8192 context length english model`          | `johnnywalee/serverless-vectorizer:latest-nomic-ai/nomic-embed-text-v1.5`                      |
| Mxbai-Embed-Large-v1                         | `mixedbread-ai/mxbai-embed-large-v1`                  | 1024      | `Large English embedding model from MixedBreed.ai`                         | `johnnywalee/serverless-vectorizer:latest-mixedbread-ai/mxbai-embed-large-v1`                  |
| Embeddinggemma-300m-ONNX                     | `onnx-community/embeddinggemma-300m-ONNX`             | 768       | `EmbeddingGemma is a 300M parameter from Google`                           | `johnnywalee/serverless-vectorizer:latest-onnx-community/embeddinggemma-300m-ONNX`             |
| Snowflake-Arctic-Embed-Xs                    | `snowflake/snowflake-arctic-embed-xs`                 | 384       | `Quantized Snowflake Arctic embed model, xs`                               | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-xs`                 |
| Mxbai-Embed-Large-v1                         | `mixedbread-ai/mxbai-embed-large-v1`                  | 1024      | `Quantized Large English embedding model from MixedBreed.ai`               | `johnnywalee/serverless-vectorizer:latest-mixedbread-ai/mxbai-embed-large-v1`                  |
| Snowflake-Arctic-Embed-M                     | `Snowflake/snowflake-arctic-embed-m`                  | 768       | `Quantized Snowflake Arctic embed model, medium`                           | `johnnywalee/serverless-vectorizer:latest-Snowflake/snowflake-arctic-embed-m`                  |
| Snowflake-Arctic-Embed-L                     | `snowflake/snowflake-arctic-embed-l`                  | 1024      | `Snowflake Arctic embed model, large`                                      | `johnnywalee/serverless-vectorizer:latest-snowflake/snowflake-arctic-embed-l`                  |
| Paraphrase-Multilingual-MINILM-L12-v2        | `Xenova/paraphrase-multilingual-MiniLM-L12-v2`        | 384       | `Multi-lingual model`                                                      | `johnnywalee/serverless-vectorizer:latest-Xenova/paraphrase-multilingual-MiniLM-L12-v2`        |
| BGE-Large-EN-v1.5-Onnx-Q                     | `Qdrant/bge-large-en-v1.5-onnx-Q`                     | 1024      | `Quantized v1.5 release of the large English model`                        | `johnnywalee/serverless-vectorizer:latest-Qdrant/bge-large-en-v1.5-onnx-Q`                     |


## Additional Supported Models

The following models are supported by fastembed-rs and can be built using the [Building Your Own Image](#building-your-own-image) instructions below. Prebuilt images are not yet available for these models.

### Image Embedding Models

| Model | Model ID | Dimension | Description |
|-------|----------|-----------|-------------|
| Clip-ViT-B-32-Vision | `Qdrant/clip-ViT-B-32-vision` | 512 | CLIP vision encoder based on ViT-B/32 |
| Resnet50-Onnx | `Qdrant/resnet50-onnx` | 2048 | ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__. |
| Unicom-ViT-B-16 | `Qdrant/Unicom-ViT-B-16` | 768 | Unicom Unicom-ViT-B-16 from open-metric-learning |
| Unicom-ViT-B-32 | `Qdrant/Unicom-ViT-B-32` | 512 | Unicom Unicom-ViT-B-32 from open-metric-learning |
| Nomic-Embed-Vision-v1.5 | `nomic-ai/nomic-embed-vision-v1.5` | 768 | Nomic NomicEmbedVisionV15 |

### Sparse Text Embedding Models

| Model | Model ID | Dimension | Description |
|-------|----------|-----------|-------------|
| Splade_PP_en_v1 | `Qdrant/Splade_PP_en_v1` | - | Splade sparse vector model for commercial use, v1 |
| BGE-M3 | `BAAI/bge-m3` | - | BGE-M3 sparse embedding model with 8192 context, supports 100+ languages |

### Reranking Models

| Model | Model ID | Dimension | Description |
|-------|----------|-----------|-------------|
| BGE-Reranker-Base | `BAAI/bge-reranker-base` | - | reranker model for English and Chinese |
| BGE-Reranker-v2-M3 | `rozgo/bge-reranker-v2-m3` | - | reranker model for multilingual |
| JINA-Reranker-v1-Turbo-EN | `jinaai/jina-reranker-v1-turbo-en` | - | reranker model for English |
| JINA-Reranker-v2-Base-Multilingual | `jinaai/jina-reranker-v2-base-multilingual` | - | reranker model for multilingual |




## Building Your Own Image

The build process uses a two-stage approach:

1. **Base image** - Contains the Lambda runtime and embedding binaries
2. **Variant image** - Extends the base image with a pre-loaded model for fast cold starts

### Option 1: Use Pre-built Base Image (Recommended)

Use the pre-built base image from Docker Hub to skip the base build step:

```bash
# Build a model variant using the pre-built base image
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=Xenova/all-MiniLM-L12-v2 \
  -f Dockerfile.variant \
  -t my-vectorizer:minilm .
```

### Option 2: Build Everything from Source

#### Step 1: Build the Base Image

```bash
docker build -t serverless-vectorizer:base .
```

#### Step 2: Build a Model Variant

Use `Dockerfile.variant` with the following build arguments:

- `BASE_IMAGE` - The base image to extend (your local build or `johnnywalee/serverless-vectorizer:base-latest`)
- `MODEL_ID` - The model ID from the [Supported Models](#supported-models) table above

```bash
docker build \
  --build-arg BASE_IMAGE=serverless-vectorizer:base \
  --build-arg MODEL_ID=<MODEL_ID> \
  -f Dockerfile.variant \
  -t serverless-vectorizer:<your-tag> .
```

### Build Examples

```bash
# BGE-Small (384 dimensions, English)
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=Xenova/bge-small-en-v1.5 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:bge-small .

# BGE-M3 (1024 dimensions, Multilingual)
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=BAAI/bge-m3 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:bge-m3 .

# Snowflake Arctic Embed Large (1024 dimensions)
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=snowflake/snowflake-arctic-embed-l \
  -f Dockerfile.variant \
  -t serverless-vectorizer:arctic-l .

# Multilingual E5 Large (1024 dimensions)
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=Qdrant/multilingual-e5-large-onnx \
  -f Dockerfile.variant \
  -t serverless-vectorizer:e5-large .

# All-MiniLM (384 dimensions, lightweight)
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=Xenova/all-MiniLM-L6-v2 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:minilm .
```

### List Available Models

Use the included CLI tool to list all supported models:

```bash
# Build and run the list-models tool
cargo run --bin list-models

# Output as markdown table
cargo run --bin list-models -- -f markdown

# Output as JSON
cargo run --bin list-models -- -f json

# List all model categories (text, image, sparse, rerank)
cargo run --bin list-models -- -c all
```





## Lambda API Reference

The Lambda automatically detects the model type from the `MODEL_ID` environment variable and routes requests accordingly. Each model type has its own request/response format.

### Model Type Auto-Detection

| MODEL_ID Pattern | Model Type | Use Case |
|-----------------|------------|----------|
| Text embedding models | `text` | Semantic search, similarity |
| `Qdrant/clip-ViT-B-32-vision`, etc. | `image` | Image similarity, visual search |
| `Qdrant/Splade_PP_en_v1`, etc. | `sparse` | Hybrid search, keyword matching |
| `BAAI/bge-reranker-*`, etc. | `rerank` | Re-ranking search results |

---

## Text Embeddings

Generate dense vector embeddings for text. Default model type.

### Request

```json
{
  "messages": ["Hello world", "How are you?"]
}
```

Or read from S3:

```json
{
  "s3_file": "my-bucket/path/to/texts.json"
}
```

### Response

```json
{
  "embeddings": [
    [0.123, 0.456, -0.789, ...],
    [0.321, 0.654, -0.987, ...]
  ],
  "dimension": 384,
  "model_type": "text",
  "count": 2
}
```

### Examples

**Direct Lambda Invocation:**

```bash
aws lambda invoke \
  --function-name serverless-vectorizer \
  --payload '{"messages": ["Hello world", "How are you?"]}' \
  response.json
```

**Local Docker Testing:**

```bash
# Start the container
docker run -p 9000:8080 johnnywalee/serverless-vectorizer:latest-Xenova/bge-small-en-v1.5

# Send request
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Hello world", "How are you?"]}'
```

**API Gateway:**

```bash
curl -X POST https://your-api.execute-api.region.amazonaws.com/embed \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Hello world", "How are you?"]}'
```

---

## Image Embeddings

Generate dense vector embeddings for images. Requires an image embedding model (e.g., `Qdrant/clip-ViT-B-32-vision`).

### Request

Images can be provided as base64-encoded data or S3 paths:

```json
{
  "images": [
    {"base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ..."},
    {"s3_path": "my-bucket/images/photo.jpg"}
  ]
}
```

Or using `s3_images` for multiple S3 paths:

```json
{
  "s3_images": [
    "my-bucket/images/photo1.jpg",
    "my-bucket/images/photo2.png"
  ]
}
```

### Response

```json
{
  "embeddings": [
    [0.123, 0.456, -0.789, ...],
    [0.321, 0.654, -0.987, ...]
  ],
  "dimension": 512,
  "model_type": "image",
  "count": 2
}
```

### Examples

**Build Image Embedding Container:**

```bash
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=Qdrant/clip-ViT-B-32-vision \
  -f Dockerfile.variant \
  -t serverless-vectorizer:clip .
```

**Local Docker Testing:**

```bash
# Start the container
docker run -p 9000:8080 serverless-vectorizer:clip

# Send request with base64 image
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {"base64": "'"$(base64 -w 0 image.png)"'"}
    ]
  }'
```

**Lambda with S3 Images:**

```bash
aws lambda invoke \
  --function-name serverless-vectorizer-image \
  --payload '{
    "s3_images": ["my-bucket/images/photo1.jpg", "my-bucket/images/photo2.jpg"]
  }' \
  response.json
```

---

## Sparse Embeddings

Generate sparse vector embeddings for text using SPLADE models. Useful for hybrid search combining dense and sparse vectors.

### Request

```json
{
  "messages": ["The quick brown fox jumps over the lazy dog"]
}
```

### Response

```json
{
  "sparse_embeddings": [
    {
      "indices": [102, 456, 789, 1234, 5678],
      "values": [0.5, 0.3, 0.8, 0.2, 0.9]
    }
  ],
  "model_type": "sparse",
  "count": 1
}
```

The sparse embedding contains:
- `indices`: Token indices with non-zero weights
- `values`: Corresponding weights for each token

### Examples

**Build Sparse Embedding Container:**

```bash
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=Qdrant/Splade_PP_en_v1 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:splade .
```

**Local Docker Testing:**

```bash
# Start the container
docker run -p 9000:8080 serverless-vectorizer:splade

# Send request
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Machine learning is a subset of artificial intelligence"]}'
```

**Lambda Invocation:**

```bash
aws lambda invoke \
  --function-name serverless-vectorizer-sparse \
  --payload '{"messages": ["Machine learning is a subset of artificial intelligence"]}' \
  response.json
```

---

## Reranking

Re-rank documents based on relevance to a query. Useful for improving search results.

### Request

```json
{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of AI that enables computers to learn from data.",
    "The weather today is sunny and warm.",
    "Deep learning uses neural networks with many layers."
  ],
  "top_k": 2,
  "return_documents": true
}
```

Parameters:
- `query`: The search query
- `documents`: Array of documents to rank
- `top_k` (optional): Return only top K results
- `return_documents` (optional, default: true): Include document text in response

### Response

```json
{
  "rankings": [
    {
      "index": 0,
      "score": 0.95,
      "document": "Machine learning is a subset of AI that enables computers to learn from data."
    },
    {
      "index": 2,
      "score": 0.82,
      "document": "Deep learning uses neural networks with many layers."
    }
  ],
  "model_type": "rerank",
  "count": 2
}
```

Results are sorted by score in descending order.

### Examples

**Build Reranking Container:**

```bash
docker build \
  --build-arg BASE_IMAGE=johnnywalee/serverless-vectorizer:base-latest \
  --build-arg MODEL_ID=BAAI/bge-reranker-base \
  -f Dockerfile.variant \
  -t serverless-vectorizer:reranker .
```

**Local Docker Testing:**

```bash
# Start the container
docker run -p 9000:8080 serverless-vectorizer:reranker

# Send request
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI that enables computers to learn from data.",
      "The weather today is sunny and warm.",
      "Deep learning uses neural networks with many layers."
    ],
    "top_k": 2
  }'
```

**Lambda Invocation:**

```bash
aws lambda invoke \
  --function-name serverless-vectorizer-rerank \
  --payload '{
    "query": "Capital cities of Europe",
    "documents": [
      "Paris is the capital of France.",
      "Pizza is a popular Italian food.",
      "London is the capital of England.",
      "Berlin is the capital of Germany."
    ],
    "top_k": 3
  }' \
  response.json
```

---

## S3 Integration

All model types support reading from and writing to S3.

### Read Input from S3

**Text Embeddings:**

```bash
aws lambda invoke \
  --function-name serverless-vectorizer \
  --payload '{"s3_file": "my-bucket/path/to/texts.json"}' \
  response.json
```

The S3 file can contain:
- Plain text (embedded as single document)
- JSON array of strings (each string embedded separately)

**Image Embeddings:**

```bash
aws lambda invoke \
  --function-name serverless-vectorizer-image \
  --payload '{"s3_images": ["my-bucket/images/photo1.jpg", "my-bucket/images/photo2.png"]}' \
  response.json
```

### Save Output to S3

Save embeddings directly to S3 (text and image models only):

```bash
aws lambda invoke \
  --function-name serverless-vectorizer \
  --payload '{
    "messages": ["Hello world"],
    "save_to_s3": {
      "bucket": "my-output-bucket",
      "key": "embeddings/output.json"
    }
  }' \
  response.json
```

**Response includes S3 location:**

```json
{
  "embeddings": [[0.123, 0.456, ...]],
  "dimension": 384,
  "model_type": "text",
  "count": 1,
  "s3_location": "s3://my-output-bucket/embeddings/output.json"
}
```

### Full Pipeline: S3 to S3

Read from S3 and save results to S3:

```bash
aws lambda invoke \
  --function-name serverless-vectorizer \
  --payload '{
    "s3_file": "input-bucket/documents.json",
    "save_to_s3": {
      "bucket": "output-bucket",
      "key": "embeddings/result.json"
    }
  }' \
  response.json
```

---

## Complete Request Schema

```json
{
  // === Text Embedding Input ===
  "messages": ["text1", "text2"],           // Direct text input
  "s3_file": "bucket/key",                  // OR read text from S3

  // === Image Embedding Input ===
  "images": [                               // Image input array
    {"base64": "..."},                      // Base64 encoded image
    {"s3_path": "bucket/key"}               // OR S3 path to image
  ],
  "s3_images": ["bucket/key1", "bucket/key2"], // OR S3 paths array

  // === Reranking Input ===
  "query": "search query",                  // Query for reranking
  "documents": ["doc1", "doc2"],            // Documents to rank
  "top_k": 5,                               // Return top K results (optional)
  "return_documents": true,                 // Include docs in response (optional)

  // === Output Options ===
  "save_to_s3": {                           // Save results to S3 (optional)
    "bucket": "bucket-name",
    "key": "path/to/output.json"
  }
}
```

## Complete Response Schema

**Text/Image Embeddings:**

```json
{
  "embeddings": [[...], [...]],             // Dense embedding vectors
  "dimension": 384,                         // Vector dimension
  "model_type": "text",                     // "text" or "image"
  "count": 2,                               // Number of embeddings
  "s3_location": "s3://..."                 // If save_to_s3 was used
}
```

**Sparse Embeddings:**

```json
{
  "sparse_embeddings": [
    {"indices": [...], "values": [...]}
  ],
  "model_type": "sparse",
  "count": 1
}
```

**Reranking:**

```json
{
  "rankings": [
    {"index": 0, "score": 0.95, "document": "..."}
  ],
  "model_type": "rerank",
  "count": 2
}
```

## Development

### Prerequisites

- Rust 1.70+
- Docker & Docker Compose
- AWS CLI (for local testing)

### Running Tests

```bash
# Run unit tests
cargo test --test unit_tests --features aws

# Start LocalStack for integration tests
docker-compose up -d

# Run all integration tests (can run in parallel - each has its own MODEL_ID)
cargo test --features aws --test integration_text_tests &
cargo test --features aws --test integration_image_tests &
cargo test --features aws --test integration_sparse_tests &
cargo test --features aws --test integration_rerank_tests &
wait

# Or run specific model type tests
cargo test --features aws --test integration_text_tests    # Text embeddings
cargo test --features aws --test integration_image_tests   # Image embeddings
cargo test --features aws --test integration_sparse_tests  # Sparse embeddings
cargo test --features aws --test integration_rerank_tests  # Reranking

# Stop LocalStack
docker-compose down
```

### Project Structure

```
.
├── src/
│   ├── main.rs              # Lambda entry point
│   ├── lambda.rs            # Lambda handler and AWS integration
│   ├── lib.rs               # Library exports
│   └── core/
│       ├── model.rs         # Model definitions and registry
│       ├── embeddings.rs    # Embedding services (text, image, sparse, rerank)
│       ├── image_utils.rs   # Image loading utilities
│       └── ...
├── tests/
│   ├── unit_tests.rs                 # Unit tests
│   ├── integration_text_tests.rs     # Text embedding integration tests
│   ├── integration_image_tests.rs    # Image embedding integration tests
│   ├── integration_sparse_tests.rs   # Sparse embedding integration tests
│   └── integration_rerank_tests.rs   # Reranking integration tests
├── Dockerfile               # Base image
├── Dockerfile.variant       # Model-specific variant builder
└── docker-compose.yaml      # LocalStack for testing
```

## Deployment

Push to ECR and create Lambda function:

```bash
# Tag and push
aws ecr get-login-password --region us-east-1 | docker login --name AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag serverless-vectorizer:bge-small 123456789.dkr.ecr.us-east-1.amazonaws.com/serverless-vectorizer:bge-small
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/serverless-vectorizer:bge-small

# Create Lambda (first time)
aws lambda create-function \
  --function-name serverless-vectorizer \
  --package-type Image \
  --code ImageUri=123456789.dkr.ecr.us-east-1.amazonaws.com/serverless-vectorizer:bge-small \
  --role arn:aws:iam::123456789:role/lambda-execution-role \
  --memory-size 1024 \
  --timeout 30

# Update Lambda (subsequent deploys)
aws lambda update-function-code \
  --function-name serverless-vectorizer \
  --image-uri 123456789.dkr.ecr.us-east-1.amazonaws.com/serverless-vectorizer:bge-small
```

## Acknowledgments

This project is powered by [fastembed-rs](https://github.com/Anush008/fastembed-rs), a Rust library for fast, lightweight embedding generation. fastembed-rs supports:

- **Text Embeddings** - Dense vector representations for semantic search and similarity
- **Image Embeddings** - Vision encoders like CLIP and ResNet for image similarity
- **Sparse Text Embeddings** - SPLADE models for hybrid search
- **Reranking Models** - Cross-encoder models for result reranking

## License

MIT

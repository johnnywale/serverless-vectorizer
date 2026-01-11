# Serverless Vectorizer

[![CI](https://github.com/johnnywale/serverless-vectorizer/actions/workflows/test-and-tag.yml/badge.svg)](https://github.com/johnnywale/serverless-vectorizer/actions/workflows/ci.yml)
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





## Usage

The Lambda supports two invocation methods:

### Direct Lambda Invocation

```bash
aws lambda invoke \
  --function-name serverless-vectorizer \
  --payload '{"messages": ["Hello world", "How are you?"]}' \
  response.json
```

**Response:**

```json
{
  "embeddings": [
    [
      0.123,
      0.456,
      ...
    ],
    [
      0.789,
      0.012,
      ...
    ]
  ],
  "dimension": 384
}
```

### API Gateway (POST /embed)

```bash
curl -X POST https://your-api.execute-api.region.amazonaws.com/embed \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Hello world", "How are you?"]}'
```

## S3 Integration

### Read Input from S3

Instead of passing messages directly, read text from an S3 file:

```bash
aws lambda invoke \
  --function-name serverless-vectorizer \
  --payload '{"s3_file": "my-bucket/path/to/input.txt"}' \
  response.json
```

The S3 file can contain:

- Plain text (embedded as single document)
- JSON array of strings (each string embedded separately)

### Save Output to S3

Save embeddings directly to S3:

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
  "embeddings": [
    [
      0.123,
      0.456,
      ...
    ]
  ],
  "dimension": 384,
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

## Request Schema

```json
{
  "messages": [
    "text1",
    "text2"
  ],
  // Direct text input (array of strings)
  "s3_file": "bucket/key",
  // OR read input from S3
  "save_to_s3": {
    // Optional: save embeddings to S3
    "bucket": "bucket-name",
    "key": "path/to/output.json"
  }
}
```

Either `messages` or `s3_file` must be provided. `save_to_s3` is optional.

## Response Schema

```json
{
  "embeddings": [
    [
      ...
    ],
    [
      ...
    ]
  ],
  // Array of embedding vectors
  "dimension": 384,
  // Vector dimension
  "s3_location": "s3://..."
  // Only present if save_to_s3 was used
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

# Run integration tests
AWS_ENDPOINT_URL=http://localhost:4566 \
AWS_ACCESS_KEY_ID=test \
AWS_SECRET_ACCESS_KEY=test \
AWS_DEFAULT_REGION=us-east-1 \
cargo test --test integration_tests --features aws -- --test-threads=1

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
│       ├── embeddings.rs    # Embedding service
│       └── ...
├── tests/
│   ├── unit_tests.rs
│   └── integration_tests.rs
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

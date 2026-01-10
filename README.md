# Serverless Vectorizer

AWS Lambda container image for generating text embeddings. Models are pre-loaded into Docker images for fast cold starts - one image per model variant.

## Supported Models

| Model | ID | Dimension | Language |
|-------|-----|-----------|----------|
| BGE-Small-EN-v1.5 | `bge-small-en-v1.5` | 384 | English |
| BGE-Base-EN-v1.5 | `bge-base-en-v1.5` | 768 | English |
| BGE-Large-EN-v1.5 | `bge-large-en-v1.5` | 1024 | English |
| Multilingual-E5-Large | `multilingual-e5-large` | 1024 | Multilingual |
| All-MpNet-Base-v2 | `all-mpnet-base-v2` | 768 | English |

All models support a maximum of 512 tokens per input text.

## Building

### Base Image

```bash
docker build -t serverless-vectorizer:base .
```

### Model-Specific Variants

Each model variant bakes the model files into the image for faster Lambda cold starts:

```bash
# Build BGE-Small variant
docker build \
  --build-arg BASE_IMAGE=serverless-vectorizer:base \
  --build-arg VARIANT=bge-small \
  --build-arg MODEL_TYPE=bge-small-en-v1.5 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:bge-small .

# Build BGE-Base variant
docker build \
  --build-arg BASE_IMAGE=serverless-vectorizer:base \
  --build-arg VARIANT=bge-base \
  --build-arg MODEL_TYPE=bge-base-en-v1.5 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:bge-base .

# Build BGE-Large variant
docker build \
  --build-arg BASE_IMAGE=serverless-vectorizer:base \
  --build-arg VARIANT=bge-large \
  --build-arg MODEL_TYPE=bge-large-en-v1.5 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:bge-large .

# Build Multilingual-E5-Large variant
docker build \
  --build-arg BASE_IMAGE=serverless-vectorizer:base \
  --build-arg VARIANT=e5-large \
  --build-arg MODEL_TYPE=multilingual-e5-large \
  -f Dockerfile.variant \
  -t serverless-vectorizer:e5-large .

# Build All-MpNet variant
docker build \
  --build-arg BASE_IMAGE=serverless-vectorizer:base \
  --build-arg VARIANT=mpnet \
  --build-arg MODEL_TYPE=all-mpnet-base-v2 \
  -f Dockerfile.variant \
  -t serverless-vectorizer:mpnet .
```

## Configuration

Set the model via environment variable in your Lambda configuration:

```bash
EMBEDDING_MODEL=bge-small-en-v1.5
```

If not specified, defaults to `bge-small-en-v1.5`.

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
  "embeddings": [[0.123, 0.456, ...], [0.789, 0.012, ...]],
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
  "embeddings": [[0.123, 0.456, ...]],
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
  "messages": ["text1", "text2"],     // Direct text input (array of strings)
  "s3_file": "bucket/key",            // OR read input from S3
  "save_to_s3": {                     // Optional: save embeddings to S3
    "bucket": "bucket-name",
    "key": "path/to/output.json"
  }
}
```

Either `messages` or `s3_file` must be provided. `save_to_s3` is optional.

## Response Schema

```json
{
  "embeddings": [[...], [...]],       // Array of embedding vectors
  "dimension": 384,                   // Vector dimension
  "s3_location": "s3://..."           // Only present if save_to_s3 was used
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
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
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

## License

MIT

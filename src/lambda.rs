// Lambda-specific handler and AWS integration

use crate::core::model::{ModelCategory, ModelRegistry};
use crate::core::{
    EmbeddingService, ImageEmbeddingService, ImageInput, RerankService, SparseEmbeddingService,
};
use aws_config;
use aws_sdk_s3 as s3;
use lambda_runtime::{Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::sync::LazyLock;

// ============================================================================
// Request Types
// ============================================================================

/// Lambda request structure - supports all model types
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Request {
    // Text embedding input (existing)
    pub messages: Option<Vec<String>>,

    // Image embedding input (new)
    pub images: Option<Vec<ImageInputRequest>>,

    // Reranking input (new)
    pub query: Option<String>,
    pub documents: Option<Vec<String>>,

    // S3 input (existing, extended for images)
    pub s3_file: Option<String>,
    pub s3_images: Option<Vec<String>>,

    // S3 output (existing)
    pub save_to_s3: Option<SaveConfig>,

    // Optional top_k for reranking
    pub top_k: Option<usize>,

    // Include documents in rerank response
    pub return_documents: Option<bool>,
}

/// Image input for request - base64 or S3 path
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ImageInputRequest {
    pub base64: Option<String>,
    pub s3_path: Option<String>,
}

/// S3 save configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SaveConfig {
    pub bucket: String,
    pub key: String,
}

// ============================================================================
// Response Types
// ============================================================================

/// Lambda response structure - supports all model types
#[derive(Serialize, Deserialize, Debug)]
pub struct Response {
    // Dense embeddings (text/image)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<Vec<Vec<f32>>>,

    // Sparse embeddings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse_embeddings: Option<Vec<SparseEmbeddingResponse>>,

    // Rerank results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rankings: Option<Vec<RerankResponse>>,

    // Metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<usize>,

    pub model_type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub s3_location: Option<String>,
}

/// Sparse embedding response format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SparseEmbeddingResponse {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

/// Rerank response format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RerankResponse {
    pub index: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

/// API Gateway response wrapper
#[derive(Serialize, Deserialize)]
pub struct ApiGatewayResponse {
    #[serde(rename = "statusCode")]
    pub status_code: u16,
    pub headers: std::collections::HashMap<String, String>,
    pub body: String,
}

// ============================================================================
// Global Services (initialized once per cold start)
// ============================================================================

// Text embedding service
static TEXT_SERVICE: LazyLock<EmbeddingService> = LazyLock::new(|| EmbeddingService::new());

// Image embedding service
static IMAGE_SERVICE: LazyLock<ImageEmbeddingService> =
    LazyLock::new(|| ImageEmbeddingService::new());

// Sparse embedding service
static SPARSE_SERVICE: LazyLock<SparseEmbeddingService> =
    LazyLock::new(|| SparseEmbeddingService::new());

// Rerank service
static RERANK_SERVICE: LazyLock<RerankService> = LazyLock::new(|| RerankService::new());

// ============================================================================
// Model Detection
// ============================================================================

/// Get the model ID from environment
fn get_model_id() -> String {
    env::var("MODEL_ID").unwrap_or_else(|_| "Xenova/bge-small-en-v1.5".to_string())
}

/// Detect model category from MODEL_ID environment variable
fn detect_model_category(model_id: &str) -> ModelCategory {
    if ModelRegistry::find_text_model(model_id).is_some() {
        ModelCategory::TextEmbedding
    } else if ModelRegistry::find_image_model(model_id).is_some() {
        ModelCategory::ImageEmbedding
    } else if ModelRegistry::find_sparse_model(model_id).is_some() {
        ModelCategory::SparseTextEmbedding
    } else if ModelRegistry::find_rerank_model(model_id).is_some() {
        ModelCategory::TextRerank
    } else {
        // Default to text embedding for unknown models
        eprintln!(
            "Warning: Unknown model '{}', defaulting to TextEmbedding category",
            model_id
        );
        ModelCategory::TextEmbedding
    }
}

async fn read_from_s3(s3_client: &s3::Client, s3_path: &str) -> Result<String, Error> {
    let parts: Vec<&str> = s3_path.splitn(2, '/').collect();
    if parts.len() != 2 {
        return Err("Invalid S3 path format. Expected: bucket/key".into());
    }

    let resp = s3_client
        .get_object()
        .bucket(parts[0])
        .key(parts[1])
        .send()
        .await?;

    let data = resp.body.collect().await?;
    Ok(String::from_utf8(data.to_vec())?)
}

async fn save_to_s3(
    s3_client: &s3::Client,
    bucket: &str,
    key: &str,
    embeddings: &[Vec<f32>],
) -> Result<(), Error> {
    let json = serde_json::to_string(&embeddings)?;

    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(json.into_bytes().into())
        .content_type("application/json")
        .send()
        .await?;

    Ok(())
}

fn is_api_gateway_request(event: &LambdaEvent<Value>) -> bool {
    event.payload.get("requestContext").is_some()
        || event.payload.get("httpMethod").is_some()
        || event.payload.get("headers").is_some()
}

fn parse_api_gateway_body(event: &LambdaEvent<Value>) -> Result<Request, Error> {
    let body = event
        .payload
        .get("body")
        .and_then(|b| b.as_str())
        .ok_or("Missing body in API Gateway request")?;

    serde_json::from_str(body).map_err(|e| format!("Failed to parse body: {}", e).into())
}

fn create_api_gateway_response(status_code: u16, body: Value) -> ApiGatewayResponse {
    let mut headers = std::collections::HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());

    ApiGatewayResponse {
        status_code,
        headers,
        body: body.to_string(),
    }
}

// ============================================================================
// Request Processing - Routes to appropriate handler based on model type
// ============================================================================

async fn process_request(request: Request) -> Result<Response, Error> {
    let model_id = get_model_id();
    let category = detect_model_category(&model_id);

    match category {
        ModelCategory::TextEmbedding => process_text_embedding(request, &model_id).await,
        ModelCategory::ImageEmbedding => process_image_embedding(request, &model_id).await,
        ModelCategory::SparseTextEmbedding => process_sparse_embedding(request, &model_id).await,
        ModelCategory::TextRerank => process_rerank(request, &model_id).await,
    }
}

// ============================================================================
// Text Embedding Handler
// ============================================================================

async fn process_text_embedding(request: Request, model_id: &str) -> Result<Response, Error> {
    // Get texts to embed
    let texts = if let Some(messages) = &request.messages {
        if messages.is_empty() {
            return Err("Messages array cannot be empty".into());
        }
        messages.clone()
    } else if let Some(s3_path) = &request.s3_file {
        let config = aws_config::load_from_env().await;
        let s3_client = s3::Client::new(&config);
        let content = read_from_s3(&s3_client, s3_path).await?;

        // Parse S3 content - could be a single text or JSON array of texts
        match serde_json::from_str::<Vec<String>>(&content) {
            Ok(texts) => texts,
            Err(_) => vec![content],
        }
    } else {
        return Err("Either 'messages' or 's3_file' must be provided for text embedding".into());
    };

    // Get the fastembed model enum
    let model = ModelRegistry::find_text_model(model_id)
        .ok_or_else(|| format!("Unknown text model: {}", model_id))?;

    // Generate embeddings using shared service
    let embeddings = TEXT_SERVICE
        .embed_with_model(texts, model)
        .map_err(|e| format!("Text embedding failed: {}", e))?;

    let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
    let count = embeddings.len();

    // Optionally save to S3
    let s3_location = if let Some(save_config) = &request.save_to_s3 {
        let config = aws_config::load_from_env().await;
        let s3_client = s3::Client::new(&config);

        save_to_s3(
            &s3_client,
            &save_config.bucket,
            &save_config.key,
            &embeddings,
        )
        .await?;

        Some(format!("s3://{}/{}", save_config.bucket, save_config.key))
    } else {
        None
    };

    Ok(Response {
        embeddings: Some(embeddings),
        sparse_embeddings: None,
        rankings: None,
        dimension: Some(dimension),
        model_type: "text".to_string(),
        count: Some(count),
        s3_location,
    })
}

// ============================================================================
// Image Embedding Handler
// ============================================================================

async fn process_image_embedding(request: Request, model_id: &str) -> Result<Response, Error> {
    use crate::core::image_utils::s3::load_image_bytes_async;

    // Check if we need S3 client (any S3 paths in request)
    let has_s3_paths = request.s3_images.is_some()
        || request
            .images
            .as_ref()
            .map(|imgs| imgs.iter().any(|i| i.s3_path.is_some()))
            .unwrap_or(false);

    let s3_client = if has_s3_paths {
        let config = aws_config::load_from_env().await;
        Some(s3::Client::new(&config))
    } else {
        None
    };

    // Collect image inputs
    let mut image_inputs: Vec<ImageInput> = Vec::new();

    // From direct image inputs
    if let Some(images) = &request.images {
        for img in images {
            if let Some(base64_data) = &img.base64 {
                image_inputs.push(ImageInput::from_base64(base64_data.clone()));
            } else if let Some(s3_path) = &img.s3_path {
                image_inputs.push(ImageInput::from_s3(s3_path.clone()));
            }
        }
    }

    // From S3 image paths
    if let Some(s3_images) = &request.s3_images {
        for s3_path in s3_images {
            image_inputs.push(ImageInput::from_s3(s3_path.clone()));
        }
    }

    if image_inputs.is_empty() {
        return Err("Either 'images' or 's3_images' must be provided for image embedding".into());
    }

    // Load all image bytes
    let mut image_bytes_list: Vec<Vec<u8>> = Vec::new();
    for input in &image_inputs {
        let bytes = load_image_bytes_async(input, s3_client.as_ref())
            .await
            .map_err(|e| format!("Failed to load image: {}", e))?;
        image_bytes_list.push(bytes);
    }

    // Get the fastembed model enum
    let model = ModelRegistry::find_image_model(model_id)
        .ok_or_else(|| format!("Unknown image model: {}", model_id))?;

    // Generate embeddings
    let embeddings = IMAGE_SERVICE
        .embed_images_with_model(&image_bytes_list, model)
        .map_err(|e| format!("Image embedding failed: {}", e))?;

    let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
    let count = embeddings.len();

    // Optionally save to S3
    let s3_location = if let Some(save_config) = &request.save_to_s3 {
        let client = if let Some(ref c) = s3_client {
            c.clone()
        } else {
            let config = aws_config::load_from_env().await;
            s3::Client::new(&config)
        };

        save_to_s3(&client, &save_config.bucket, &save_config.key, &embeddings).await?;

        Some(format!("s3://{}/{}", save_config.bucket, save_config.key))
    } else {
        None
    };

    Ok(Response {
        embeddings: Some(embeddings),
        sparse_embeddings: None,
        rankings: None,
        dimension: Some(dimension),
        model_type: "image".to_string(),
        count: Some(count),
        s3_location,
    })
}

// ============================================================================
// Sparse Text Embedding Handler
// ============================================================================

async fn process_sparse_embedding(request: Request, model_id: &str) -> Result<Response, Error> {
    // Get texts to embed
    let texts = if let Some(messages) = &request.messages {
        if messages.is_empty() {
            return Err("Messages array cannot be empty".into());
        }
        messages.clone()
    } else if let Some(s3_path) = &request.s3_file {
        let config = aws_config::load_from_env().await;
        let s3_client = s3::Client::new(&config);
        let content = read_from_s3(&s3_client, s3_path).await?;

        match serde_json::from_str::<Vec<String>>(&content) {
            Ok(texts) => texts,
            Err(_) => vec![content],
        }
    } else {
        return Err("Either 'messages' or 's3_file' must be provided for sparse embedding".into());
    };

    // Get the fastembed model enum
    let model = ModelRegistry::find_sparse_model(model_id)
        .ok_or_else(|| format!("Unknown sparse model: {}", model_id))?;

    // Generate sparse embeddings
    let sparse_embeddings = SPARSE_SERVICE
        .embed_with_model(texts, model)
        .map_err(|e| format!("Sparse embedding failed: {}", e))?;

    let count = sparse_embeddings.len();

    // Convert to response format
    let sparse_responses: Vec<SparseEmbeddingResponse> = sparse_embeddings
        .into_iter()
        .map(|se| SparseEmbeddingResponse {
            indices: se.indices,
            values: se.values,
        })
        .collect();

    Ok(Response {
        embeddings: None,
        sparse_embeddings: Some(sparse_responses),
        rankings: None,
        dimension: None,
        model_type: "sparse".to_string(),
        count: Some(count),
        s3_location: None,
    })
}

// ============================================================================
// Reranking Handler
// ============================================================================

async fn process_rerank(request: Request, model_id: &str) -> Result<Response, Error> {
    let query = request
        .query
        .ok_or("'query' is required for reranking")?;

    let documents = request
        .documents
        .ok_or("'documents' is required for reranking")?;

    if documents.is_empty() {
        return Err("'documents' array cannot be empty".into());
    }

    let return_documents = request.return_documents.unwrap_or(true);

    // Get the fastembed model enum
    let model = ModelRegistry::find_rerank_model(model_id)
        .ok_or_else(|| format!("Unknown rerank model: {}", model_id))?;

    // Perform reranking
    let results = RERANK_SERVICE
        .rerank_with_model(&query, documents.clone(), return_documents, model)
        .map_err(|e| format!("Reranking failed: {}", e))?;

    // Apply top_k if specified
    let results = if let Some(top_k) = request.top_k {
        results.into_iter().take(top_k).collect()
    } else {
        results
    };

    let count = results.len();

    // Convert to response format
    let rankings: Vec<RerankResponse> = results
        .into_iter()
        .map(|r| RerankResponse {
            index: r.index,
            score: r.score,
            document: r.document,
        })
        .collect();

    Ok(Response {
        embeddings: None,
        sparse_embeddings: None,
        rankings: Some(rankings),
        dimension: None,
        model_type: "rerank".to_string(),
        count: Some(count),
        s3_location: None,
    })
}

/// Main Lambda handler
pub async fn handler(event: LambdaEvent<Value>) -> Result<Value, Error> {
    if is_api_gateway_request(&event) {
        let request = match parse_api_gateway_body(&event) {
            Ok(req) => req,
            Err(e) => {
                let error_response = create_api_gateway_response(
                    400,
                    json!({
                        "error": format!("Bad Request: {}", e)
                    }),
                );
                return Ok(serde_json::to_value(error_response)?);
            }
        };

        match process_request(request).await {
            Ok(response) => {
                let api_response =
                    create_api_gateway_response(200, serde_json::to_value(response)?);
                Ok(serde_json::to_value(api_response)?)
            }
            Err(e) => {
                let error_response = create_api_gateway_response(
                    500,
                    json!({
                        "error": format!("Internal Server Error: {}", e)
                    }),
                );
                Ok(serde_json::to_value(error_response)?)
            }
        }
    } else {
        let request: Request = serde_json::from_value(event.payload)?;
        let response = process_request(request).await?;
        Ok(serde_json::to_value(response)?)
    }
}

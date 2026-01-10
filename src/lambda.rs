
// Lambda-specific handler and AWS integration

use crate::core::{EmbeddingService, ModelType};
use aws_config;
use aws_sdk_s3 as s3;
use lambda_runtime::{Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::sync::LazyLock;

/// Lambda request structure
#[derive(Serialize, Deserialize, Clone)]
pub struct Request {
    pub messages: Option<Vec<String>>,
    pub s3_file: Option<String>,
    pub save_to_s3: Option<SaveConfig>,
}

/// S3 save configuration
#[derive(Serialize, Deserialize, Clone)]
pub struct SaveConfig {
    pub bucket: String,
    pub key: String,
}

/// Lambda response structure
#[derive(Serialize, Deserialize, Debug)]
pub struct Response {
    pub embeddings: Vec<Vec<f32>>,
    pub dimension: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub s3_location: Option<String>,
}

/// API Gateway response wrapper
#[derive(Serialize, Deserialize)]
pub struct ApiGatewayResponse {
    #[serde(rename = "statusCode")]
    pub status_code: u16,
    pub headers: std::collections::HashMap<String, String>,
    pub body: String,
}

// Global embedding service for Lambda (initialized once per cold start)
static EMBEDDING_SERVICE: LazyLock<EmbeddingService> = LazyLock::new(|| {
    EmbeddingService::new()
});

/// Get the model type from environment or default
fn get_model_type() -> ModelType {
    let model_str = env::var("EMBEDDING_MODEL")
        .unwrap_or_else(|_| "bge-small-en-v1.5".to_string());

    ModelType::from_str(&model_str).unwrap_or_else(|| {
        eprintln!(
            "Warning: Unknown model '{}', falling back to BGE-Small",
            model_str
        );
        ModelType::default()
    })
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

async fn process_request(request: Request) -> Result<Response, Error> {
    let model_type = get_model_type();

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
        return Err("Either 'messages' or 's3_file' must be provided".into());
    };

    // Generate embeddings using shared service
    let embeddings = EMBEDDING_SERVICE
        .embed(texts, model_type)
        .map_err(|e| format!("Embedding failed: {}", e))?;

    let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);

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
        embeddings,
        dimension,
        s3_location,
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

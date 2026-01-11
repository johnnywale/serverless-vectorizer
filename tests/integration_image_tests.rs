// tests/integration_image_tests.rs
//
// Image embedding integration tests
// MODEL_ID is set to an image embedding model

use aws_config::{BehaviorVersion, Region};
use aws_sdk_s3 as s3;
use aws_sdk_s3::config::Credentials;
use base64::{engine::general_purpose::STANDARD, Engine};
use lambda_runtime::{Context, LambdaEvent};
use serde_json::Value;
use serverless_vectorizer::{handler, ImageInputRequest, Request, Response};

const MODEL_ID: &str = "Qdrant/clip-ViT-B-32-vision";
const TEST_BUCKET: &str = "test-images-bucket";

// Setup environment for tests
fn setup_env() {
    let endpoint_url =
        std::env::var("LOCALSTACK_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:4566".to_string());

    unsafe {
        std::env::set_var("AWS_ENDPOINT_URL", &endpoint_url);
        std::env::set_var("AWS_ACCESS_KEY_ID", "test");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("MODEL_ID", MODEL_ID);
    }

    // Print diagnostic info for CI debugging
    eprintln!("[TEST] MODEL_ID set to: {}", MODEL_ID);
    eprintln!("[TEST] AWS_ENDPOINT_URL: {}", endpoint_url);
}

async fn create_s3_client() -> s3::Client {
    let endpoint_url =
        std::env::var("LOCALSTACK_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:4566".to_string());

    let creds = Credentials::new("test", "test", None, None, "test");

    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(creds)
        .endpoint_url(&endpoint_url)
        .load()
        .await;

    let s3_config = s3::config::Builder::from(&config)
        .force_path_style(true)
        .build();

    s3::Client::from_conf(s3_config)
}

fn create_lambda_event(request: Request) -> LambdaEvent<Value> {
    let payload = serde_json::to_value(request).expect("Failed to serialize request");
    LambdaEvent {
        payload,
        context: Context::default(),
    }
}

fn create_image_request(images: Option<Vec<ImageInputRequest>>, s3_images: Option<Vec<String>>) -> Request {
    Request {
        messages: None,
        images,
        query: None,
        documents: None,
        s3_file: None,
        s3_images,
        save_to_s3: None,
        top_k: None,
        return_documents: None,
    }
}

fn parse_response(value: Value) -> Result<Response, String> {
    serde_json::from_value(value).map_err(|e| format!("Failed to parse response: {}", e))
}

async fn ensure_bucket_exists(s3_client: &s3::Client, bucket: &str) -> Result<(), Box<dyn std::error::Error>> {
    match s3_client.head_bucket().bucket(bucket).send().await {
        Ok(_) => Ok(()),
        Err(_) => {
            s3_client.create_bucket().bucket(bucket).send().await?;
            Ok(())
        }
    }
}

async fn upload_binary_to_s3(
    s3_client: &s3::Client,
    bucket: &str,
    key: &str,
    content: Vec<u8>,
    content_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content.into())
        .content_type(content_type)
        .send()
        .await?;
    Ok(())
}

// Create a minimal valid PNG image (1x1 transparent pixel)
fn create_test_png() -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        .expect("Failed to decode test PNG")
}

#[tokio::test]
async fn test_image_embedding_base64() {
    setup_env();

    let png_bytes = create_test_png();
    eprintln!("[TEST] Created test PNG with {} bytes", png_bytes.len());

    // Verify PNG magic bytes
    if png_bytes.len() >= 8 {
        eprintln!("[TEST] PNG header: {:02X?}", &png_bytes[..8]);
    }

    let base64_image = STANDARD.encode(&png_bytes);
    eprintln!("[TEST] Base64 encoded length: {}", base64_image.len());

    let request = create_image_request(
        Some(vec![ImageInputRequest {
            base64: Some(base64_image),
            s3_path: None,
        }]),
        None,
    );

    eprintln!("[TEST] Calling handler...");
    let event = create_lambda_event(request);
    let result = handler(event).await;

    match &result {
        Ok(value) => {
            eprintln!("[TEST] Handler succeeded");
            eprintln!("[TEST] Response: {}", serde_json::to_string_pretty(value).unwrap_or_default());
        },
        Err(e) => {
            eprintln!("[TEST] Handler error: {:?}", e);
            eprintln!("[TEST] Error details: {}", e);
        }
    }
    assert!(result.is_ok(), "Handler should succeed for image embedding: {:?}", result.err());

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "image", "Should be image model type, got: {}", response.model_type);
    assert!(response.embeddings.is_some(), "Should have embeddings");

    let embeddings = response.embeddings.unwrap();
    assert_eq!(embeddings.len(), 1, "Should have one embedding");
    assert!(!embeddings[0].is_empty(), "Embedding should not be empty");
    assert_eq!(response.dimension.unwrap_or(0), 512, "CLIP ViT-B/32 has 512 dimensions");
}

#[tokio::test]
async fn test_image_embedding_from_s3() {
    setup_env();
    let s3_client = create_s3_client().await;
    ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

    let png_bytes = create_test_png();
    let s3_key = "test-images/test.png";

    upload_binary_to_s3(&s3_client, TEST_BUCKET, s3_key, png_bytes, "image/png")
        .await
        .unwrap();

    let request = create_image_request(
        None,
        Some(vec![format!("{}/{}", TEST_BUCKET, s3_key)]),
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for S3 image");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "image");
    assert!(response.embeddings.is_some());
}

#[tokio::test]
async fn test_image_embedding_multiple_base64() {
    setup_env();

    let png_bytes = create_test_png();
    let base64_image = STANDARD.encode(&png_bytes);

    let request = create_image_request(
        Some(vec![
            ImageInputRequest {
                base64: Some(base64_image.clone()),
                s3_path: None,
            },
            ImageInputRequest {
                base64: Some(base64_image),
                s3_path: None,
            },
        ]),
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for multiple images");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "image");

    let embeddings = response.embeddings.unwrap();
    assert_eq!(embeddings.len(), 2, "Should have 2 embeddings");
}

#[tokio::test]
async fn test_image_embedding_mixed_input() {
    setup_env();
    let s3_client = create_s3_client().await;
    ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

    let png_bytes = create_test_png();
    let s3_key = "test-images/mixed-test.png";

    upload_binary_to_s3(&s3_client, TEST_BUCKET, s3_key, png_bytes.clone(), "image/png")
        .await
        .unwrap();

    let base64_image = STANDARD.encode(&png_bytes);

    let request = Request {
        messages: None,
        images: Some(vec![ImageInputRequest {
            base64: Some(base64_image),
            s3_path: None,
        }]),
        query: None,
        documents: None,
        s3_file: None,
        s3_images: Some(vec![format!("{}/{}", TEST_BUCKET, s3_key)]),
        save_to_s3: None,
        top_k: None,
        return_documents: None,
    };

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for mixed input");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "image");

    let embeddings = response.embeddings.unwrap();
    assert_eq!(embeddings.len(), 2, "Should have embeddings for both images");
}

#[tokio::test]
async fn test_image_embedding_empty_input() {
    setup_env();

    let request = create_image_request(None, None);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_err(), "Should fail with no images");
}

#[tokio::test]
async fn test_image_embedding_consistency() {
    setup_env();

    let png_bytes = create_test_png();
    let base64_image = STANDARD.encode(&png_bytes);

    let request1 = create_image_request(
        Some(vec![ImageInputRequest {
            base64: Some(base64_image.clone()),
            s3_path: None,
        }]),
        None,
    );

    let request2 = create_image_request(
        Some(vec![ImageInputRequest {
            base64: Some(base64_image),
            s3_path: None,
        }]),
        None,
    );

    let event1 = create_lambda_event(request1);
    let event2 = create_lambda_event(request2);

    let result1 = handler(event1).await;
    let result2 = handler(event2).await;

    assert!(result1.is_ok() && result2.is_ok(), "Both requests should succeed");

    let response1 = parse_response(result1.unwrap()).unwrap();
    let response2 = parse_response(result2.unwrap()).unwrap();

    let emb1 = response1.embeddings.unwrap();
    let emb2 = response2.embeddings.unwrap();

    let match_result = emb1[0]
        .iter()
        .zip(emb2[0].iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    assert!(match_result, "Same image should produce identical embeddings");
}

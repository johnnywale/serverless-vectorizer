// tests/integration_text_tests.rs
//
// Text embedding integration tests
// MODEL_ID is set to a text embedding model

use aws_config::{BehaviorVersion, Region};
use aws_sdk_s3 as s3;
use aws_sdk_s3::config::Credentials;
use lambda_runtime::{Context, LambdaEvent};
use serde_json::Value;
use serverless_vectorizer::{handler, Request, Response};

const MODEL_ID: &str = "Xenova/bge-small-en-v1.5";
const TEST_BUCKET: &str = "test-text-embeddings-bucket";

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

fn create_text_request(messages: Option<Vec<String>>, s3_file: Option<String>) -> Request {
    Request {
        messages,
        images: None,
        query: None,
        documents: None,
        s3_file,
        s3_images: None,
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

async fn upload_text_to_s3(
    s3_client: &s3::Client,
    bucket: &str,
    key: &str,
    content: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content.as_bytes().to_vec().into())
        .send()
        .await?;
    Ok(())
}

async fn upload_json_to_s3(
    s3_client: &s3::Client,
    bucket: &str,
    key: &str,
    messages: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let json_content = serde_json::to_string(messages)?;
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(json_content.as_bytes().to_vec().into())
        .content_type("application/json")
        .send()
        .await?;
    Ok(())
}

#[tokio::test]
async fn test_text_embedding_direct_input() {
    setup_env();

    let request = create_text_request(
        Some(vec!["Hello world".to_string(), "How are you?".to_string()]),
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for text embedding");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "text", "Should be text model type");
    assert!(response.embeddings.is_some(), "Should have embeddings");

    let embeddings = response.embeddings.unwrap();
    assert_eq!(embeddings.len(), 2, "Should have 2 embeddings");
    assert_eq!(response.dimension.unwrap_or(0), 384);
}

#[tokio::test]
async fn test_text_embedding_single_text_from_s3() {
    setup_env();
    let s3_client = create_s3_client().await;
    ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

    let test_text = "Hello from S3!";
    let s3_key = "test-files/single-text.txt";

    upload_text_to_s3(&s3_client, TEST_BUCKET, s3_key, test_text)
        .await
        .unwrap();

    let request = create_text_request(None, Some(format!("{}/{}", TEST_BUCKET, s3_key)));

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed reading from S3");

    let response = parse_response(result.unwrap()).unwrap();
    let embeddings = response.embeddings.expect("Should have embeddings");
    assert_eq!(embeddings.len(), 1);
    assert_eq!(response.dimension.unwrap_or(0), 384);
}

#[tokio::test]
async fn test_text_embedding_json_array_from_s3() {
    setup_env();
    let s3_client = create_s3_client().await;
    ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

    let messages = vec![
        "First message from S3".to_string(),
        "Second message from S3".to_string(),
        "Third message from S3".to_string(),
    ];
    let s3_key = "test-files/messages-array.json";

    upload_json_to_s3(&s3_client, TEST_BUCKET, s3_key, &messages)
        .await
        .unwrap();

    let request = create_text_request(None, Some(format!("{}/{}", TEST_BUCKET, s3_key)));

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed reading JSON array from S3");

    let response = parse_response(result.unwrap()).unwrap();
    let embeddings = response.embeddings.expect("Should have embeddings");
    assert_eq!(embeddings.len(), 3);
    assert_eq!(response.dimension.unwrap_or(0), 384);
}

#[tokio::test]
async fn test_text_embedding_s3_consistency() {
    setup_env();
    let s3_client = create_s3_client().await;
    ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

    let test_text = "Consistency test message";
    let s3_key = "test-files/consistency-test.txt";

    upload_text_to_s3(&s3_client, TEST_BUCKET, s3_key, test_text)
        .await
        .unwrap();

    // Get embedding from S3
    let s3_request = create_text_request(None, Some(format!("{}/{}", TEST_BUCKET, s3_key)));
    let s3_event = create_lambda_event(s3_request);
    let s3_response = parse_response(handler(s3_event).await.unwrap()).unwrap();
    let s3_embeddings = s3_response.embeddings.expect("Should have embeddings");

    // Get embedding from direct input
    let direct_request = create_text_request(Some(vec![test_text.to_string()]), None);
    let direct_event = create_lambda_event(direct_request);
    let direct_response = parse_response(handler(direct_event).await.unwrap()).unwrap();
    let direct_embeddings = direct_response.embeddings.expect("Should have embeddings");

    // Compare embeddings
    assert_eq!(s3_embeddings[0].len(), direct_embeddings[0].len());

    let embeddings_match = s3_embeddings[0]
        .iter()
        .zip(direct_embeddings[0].iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    assert!(embeddings_match, "S3 and direct input should produce identical embeddings");
}

#[tokio::test]
async fn test_text_embedding_invalid_s3_path() {
    setup_env();

    let request = create_text_request(None, Some("nonexistent-bucket/nonexistent-file.txt".to_string()));

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_err(), "Handler should fail for invalid S3 path");
}

#[tokio::test]
async fn test_text_embedding_empty_messages() {
    setup_env();

    let request = create_text_request(Some(vec![]), None);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_err(), "Handler should fail for empty messages");
}

#[tokio::test]
async fn test_text_embedding_consistency() {
    setup_env();

    let text = "Consistent embedding test".to_string();

    let request1 = create_text_request(Some(vec![text.clone()]), None);
    let request2 = create_text_request(Some(vec![text]), None);

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

    assert!(match_result, "Same text should produce identical embeddings");
}

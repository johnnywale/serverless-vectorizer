// tests/integration_sparse_tests.rs
//
// Sparse embedding integration tests
// MODEL_ID is set to a sparse embedding model (SPLADE)

use lambda_runtime::{Context, LambdaEvent};
use serde_json::Value;
use serverless_vectorizer::{handler, Request, Response};

const MODEL_ID: &str = "Qdrant/Splade_PP_en_v1";

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
    eprintln!("[SPARSE TEST] MODEL_ID set to: {}", MODEL_ID);
}

fn create_lambda_event(request: Request) -> LambdaEvent<Value> {
    let payload = serde_json::to_value(request).expect("Failed to serialize request");
    LambdaEvent {
        payload,
        context: Context::default(),
    }
}

fn create_sparse_request(messages: Vec<String>) -> Request {
    Request {
        messages: Some(messages),
        images: None,
        query: None,
        documents: None,
        s3_file: None,
        s3_images: None,
        save_to_s3: None,
        top_k: None,
        return_documents: None,
    }
}

fn parse_response(value: Value) -> Result<Response, String> {
    serde_json::from_value(value).map_err(|e| format!("Failed to parse response: {}", e))
}

#[tokio::test]
async fn test_sparse_embedding_single_text() {
    setup_env();

    let request = create_sparse_request(vec!["The quick brown fox jumps over the lazy dog".to_string()]);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for sparse embedding");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "sparse", "Should be sparse model type");
    assert!(response.sparse_embeddings.is_some(), "Should have sparse embeddings");

    let sparse = response.sparse_embeddings.unwrap();
    assert_eq!(sparse.len(), 1);
    assert!(!sparse[0].indices.is_empty(), "Sparse embedding should have non-zero indices");
    assert_eq!(sparse[0].indices.len(), sparse[0].values.len(), "Indices and values should have same length");
}

#[tokio::test]
async fn test_sparse_embedding_batch() {
    setup_env();

    eprintln!("[SPARSE TEST] Running batch test with 3 texts");

    let request = create_sparse_request(vec![
        "Machine learning is a subset of artificial intelligence".to_string(),
        "Deep learning uses neural networks".to_string(),
        "Natural language processing handles text".to_string(),
    ]);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    match &result {
        Ok(value) => {
            eprintln!("[SPARSE TEST] Batch handler succeeded");
            eprintln!("[SPARSE TEST] Response: {}", serde_json::to_string_pretty(value).unwrap_or_default());
        },
        Err(e) => {
            eprintln!("[SPARSE TEST] Batch handler error: {:?}", e);
            eprintln!("[SPARSE TEST] Error details: {}", e);
        }
    }

    assert!(result.is_ok(), "Handler should succeed for batch sparse embedding: {:?}", result.err());

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "sparse");

    let sparse = response.sparse_embeddings.unwrap();
    assert_eq!(sparse.len(), 3, "Should have 3 sparse embeddings");

    for emb in &sparse {
        assert!(!emb.indices.is_empty(), "Each sparse embedding should have indices");
        assert_eq!(emb.indices.len(), emb.values.len());
    }
}

#[tokio::test]
async fn test_sparse_embedding_consistency() {
    setup_env();

    let text = "Consistent test for sparse embeddings".to_string();

    let request1 = create_sparse_request(vec![text.clone()]);
    let request2 = create_sparse_request(vec![text]);

    let event1 = create_lambda_event(request1);
    let event2 = create_lambda_event(request2);

    let result1 = handler(event1).await;
    let result2 = handler(event2).await;

    assert!(result1.is_ok() && result2.is_ok(), "Both requests should succeed");

    let response1 = parse_response(result1.unwrap()).unwrap();
    let response2 = parse_response(result2.unwrap()).unwrap();

    assert_eq!(response1.model_type, "sparse");
    assert_eq!(response2.model_type, "sparse");

    let sparse1 = response1.sparse_embeddings.unwrap();
    let sparse2 = response2.sparse_embeddings.unwrap();

    assert_eq!(sparse1[0].indices, sparse2[0].indices, "Indices should match");

    let values_match = sparse1[0].values.iter()
        .zip(sparse2[0].values.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(values_match, "Values should be identical");
}

#[tokio::test]
async fn test_sparse_embedding_different_texts() {
    setup_env();

    let request = create_sparse_request(vec![
        "Programming in Rust".to_string(),
        "Cooking Italian pasta".to_string(),
    ]);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "sparse");

    let sparse = response.sparse_embeddings.unwrap();

    // Different texts should have different sparse representations
    let indices_different = sparse[0].indices != sparse[1].indices;
    assert!(indices_different, "Different texts should have different sparse indices");
}

#[tokio::test]
async fn test_sparse_embedding_empty_input() {
    setup_env();

    let request = create_sparse_request(vec![]);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_err(), "Handler should fail for empty input");
}

#[tokio::test]
async fn test_sparse_embedding_long_text() {
    setup_env();

    let long_text = "This is a much longer text that contains many words. ".repeat(50);

    let request = create_sparse_request(vec![long_text]);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for long text");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "sparse");

    let sparse = response.sparse_embeddings.unwrap();
    assert_eq!(sparse.len(), 1);
    assert!(!sparse[0].indices.is_empty());
}

#[tokio::test]
async fn test_sparse_embedding_special_characters() {
    setup_env();

    let request = create_sparse_request(vec![
        "Text with special chars: @#$%^&*()".to_string(),
        "Numbers: 12345 and symbols: <>?/".to_string(),
    ]);

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should handle special characters");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "sparse");

    let sparse = response.sparse_embeddings.unwrap();
    assert_eq!(sparse.len(), 2);
}

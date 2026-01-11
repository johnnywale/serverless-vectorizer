// tests/integration_rerank_tests.rs
//
// Reranking integration tests
// MODEL_ID is set to a reranking model

use lambda_runtime::{Context, LambdaEvent};
use serde_json::Value;
use serverless_vectorizer::{handler, Request, Response};

const MODEL_ID: &str = "BAAI/bge-reranker-base";

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

fn create_lambda_event(request: Request) -> LambdaEvent<Value> {
    let payload = serde_json::to_value(request).expect("Failed to serialize request");
    LambdaEvent {
        payload,
        context: Context::default(),
    }
}

fn create_rerank_request(query: String, documents: Vec<String>, top_k: Option<usize>) -> Request {
    Request {
        messages: None,
        images: None,
        query: Some(query),
        documents: Some(documents),
        s3_file: None,
        s3_images: None,
        save_to_s3: None,
        top_k,
        return_documents: Some(true),
    }
}

fn parse_response(value: Value) -> Result<Response, String> {
    serde_json::from_value(value).map_err(|e| format!("Failed to parse response: {}", e))
}

#[tokio::test]
async fn test_rerank_basic() {
    setup_env();

    let request = create_rerank_request(
        "What is machine learning?".to_string(),
        vec![
            "Machine learning is a subset of AI that enables computers to learn from data.".to_string(),
            "The weather today is sunny and warm.".to_string(),
            "Deep learning uses neural networks with many layers.".to_string(),
        ],
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for reranking");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "rerank", "Should be rerank model type");
    assert!(response.rankings.is_some(), "Should have rankings");

    let rankings = response.rankings.unwrap();
    assert_eq!(rankings.len(), 3, "Should have rankings for all documents");

    // Results should be sorted by score (descending)
    for i in 1..rankings.len() {
        assert!(rankings[i-1].score >= rankings[i].score,
            "Rankings should be sorted by score descending");
    }

    // The ML-related document should rank higher than weather
    let ml_doc_rank = rankings.iter().position(|r| r.index == 0).unwrap();
    let weather_doc_rank = rankings.iter().position(|r| r.index == 1).unwrap();
    assert!(ml_doc_rank < weather_doc_rank,
        "ML document should rank higher than weather document");
}

#[tokio::test]
async fn test_rerank_with_top_k() {
    setup_env();

    let request = create_rerank_request(
        "Capital cities".to_string(),
        vec![
            "Paris is the capital of France.".to_string(),
            "Pizza is a popular Italian food.".to_string(),
            "London is the capital of England.".to_string(),
            "Coffee comes from beans.".to_string(),
            "Tokyo is the capital of Japan.".to_string(),
        ],
        Some(2),
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "rerank");

    let rankings = response.rankings.unwrap();
    assert_eq!(rankings.len(), 2, "Should only return top 2 results");
}

#[tokio::test]
async fn test_rerank_returns_documents() {
    setup_env();

    let docs = vec![
        "Document one about programming.".to_string(),
        "Document two about cooking.".to_string(),
    ];

    let request = create_rerank_request(
        "Programming tutorials".to_string(),
        docs.clone(),
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "rerank");

    let rankings = response.rankings.unwrap();

    for ranking in &rankings {
        assert!(ranking.document.is_some(), "Document should be included");
        let doc = ranking.document.as_ref().unwrap();
        assert!(docs.contains(doc), "Document should match original");
    }
}

#[tokio::test]
async fn test_rerank_consistency() {
    setup_env();

    let query = "Technology news".to_string();
    let documents = vec![
        "Apple released new iPhone.".to_string(),
        "Farmers harvested apples.".to_string(),
    ];

    let request1 = create_rerank_request(query.clone(), documents.clone(), None);
    let request2 = create_rerank_request(query, documents, None);

    let event1 = create_lambda_event(request1);
    let event2 = create_lambda_event(request2);

    let result1 = handler(event1).await;
    let result2 = handler(event2).await;

    assert!(result1.is_ok() && result2.is_ok(), "Both requests should succeed");

    let response1 = parse_response(result1.unwrap()).unwrap();
    let response2 = parse_response(result2.unwrap()).unwrap();

    assert_eq!(response1.model_type, "rerank");
    assert_eq!(response2.model_type, "rerank");

    let rankings1 = response1.rankings.unwrap();
    let rankings2 = response2.rankings.unwrap();

    for (r1, r2) in rankings1.iter().zip(rankings2.iter()) {
        assert_eq!(r1.index, r2.index, "Ranking order should be consistent");
        assert!((r1.score - r2.score).abs() < 1e-6, "Scores should be identical");
    }
}

#[tokio::test]
async fn test_rerank_missing_query() {
    setup_env();

    let request = Request {
        messages: None,
        images: None,
        query: None,
        documents: Some(vec!["doc1".to_string()]),
        s3_file: None,
        s3_images: None,
        save_to_s3: None,
        top_k: None,
        return_documents: None,
    };

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_err(), "Should fail with missing query");
}

#[tokio::test]
async fn test_rerank_empty_documents() {
    setup_env();

    let request = create_rerank_request(
        "Test query".to_string(),
        vec![],
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_err(), "Should fail with empty documents");
}

#[tokio::test]
async fn test_rerank_single_document() {
    setup_env();

    let request = create_rerank_request(
        "Test query".to_string(),
        vec!["Only one document here.".to_string()],
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed with single document");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "rerank");

    let rankings = response.rankings.unwrap();
    assert_eq!(rankings.len(), 1, "Should have one ranking");
    assert_eq!(rankings[0].index, 0);
}

#[tokio::test]
async fn test_rerank_large_batch() {
    setup_env();

    let documents: Vec<String> = (0..20)
        .map(|i| format!("This is document number {} with some content.", i))
        .collect();

    let request = create_rerank_request(
        "Document with number".to_string(),
        documents,
        Some(5),
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed for large batch");

    let response = parse_response(result.unwrap()).unwrap();
    assert_eq!(response.model_type, "rerank");

    let rankings = response.rankings.unwrap();
    assert_eq!(rankings.len(), 5, "Should return only top 5");
}

#[tokio::test]
async fn test_rerank_relevance_ordering() {
    setup_env();

    let request = create_rerank_request(
        "Programming languages".to_string(),
        vec![
            "Rust is a systems programming language focused on safety.".to_string(),
            "Python is popular for data science and machine learning.".to_string(),
            "The cat sat on the mat.".to_string(),
            "JavaScript runs in web browsers.".to_string(),
        ],
        None,
    );

    let event = create_lambda_event(request);
    let result = handler(event).await;

    assert!(result.is_ok(), "Handler should succeed");

    let response = parse_response(result.unwrap()).unwrap();
    let rankings = response.rankings.unwrap();

    // The cat document (index 2) should be ranked last
    let cat_rank = rankings.iter().position(|r| r.index == 2).unwrap();
    assert_eq!(cat_rank, rankings.len() - 1, "Irrelevant document should be ranked last");
}

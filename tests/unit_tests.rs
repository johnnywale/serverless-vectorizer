// tests/unit_tests.rs

use lambda_runtime::{Context, LambdaEvent};
use serde_json::{json, Value};
use serverless_vectorizer::{handler, Request, Response, SaveConfig};

// Helper function to create a test context
fn create_test_context() -> Context {
    Context::default()
}

// Helper function to create a LambdaEvent with proper Value payload
fn create_lambda_event(request: Request) -> LambdaEvent<Value> {
    let payload = serde_json::to_value(request).expect("Failed to serialize request");
    LambdaEvent {
        payload,
        context: create_test_context(),
    }
}

// Helper function to parse response from handler
fn parse_response(value: Value) -> Result<Response, String> {
    serde_json::from_value(value).map_err(|e| format!("Failed to parse response: {}", e))
}

// Helper to create a text embedding request
fn create_text_request(messages: Vec<String>) -> Request {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_text_embedding() {
        // Test basic text embedding
        let request = create_text_request(vec!["Hello, world!".to_string()]);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should succeed");

        let response = parse_response(result.unwrap()).expect("Should parse response");
        let embeddings = response.embeddings.expect("Should have embeddings");
        assert_eq!(embeddings.len(), 1, "Should have one embedding");
        assert!(embeddings[0].len() > 0, "Embedding should not be empty");
        assert_eq!(
            response.dimension.unwrap_or(0),
            embeddings[0].len(),
            "Dimension should match embedding length"
        );
        assert!(response.s3_location.is_none(), "S3 location should be None");
        assert_eq!(response.model_type, "text", "Model type should be text");

        // BGE-small-en-v1.5 produces 384-dimensional embeddings
        assert_eq!(
            response.dimension.unwrap_or(0),
            384,
            "Expected 384-dimensional embedding from BGE-small-en-v1.5"
        );
    }

    #[tokio::test]
    async fn test_multiple_messages_embedding() {
        // Test batch embedding with multiple messages
        let request = create_text_request(vec![
            "First message".to_string(),
            "Second message".to_string(),
            "Third message".to_string(),
        ]);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should succeed for multiple messages"
        );

        let response = parse_response(result.unwrap()).expect("Should parse response");
        let embeddings = response.embeddings.expect("Should have embeddings");
        assert_eq!(embeddings.len(), 3, "Should have three embeddings");
        assert_eq!(response.dimension.unwrap_or(0), 384);

        // All embeddings should be 384-dimensional
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 384);
        }

        // Different messages should produce different embeddings
        let embeddings_different = embeddings[0]
            .iter()
            .zip(embeddings[1].iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(
            embeddings_different,
            "Different messages should produce different embeddings"
        );
    }

    #[tokio::test]
    async fn test_empty_message() {
        // Test with empty string
        let request = create_text_request(vec!["".to_string()]);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle empty string");

        let response = parse_response(result.unwrap()).expect("Should parse response");
        assert_eq!(
            response.dimension.unwrap_or(0),
            384,
            "Should still produce 384-dimensional embedding"
        );
    }

    #[tokio::test]
    async fn test_long_text_embedding() {
        // Test with longer text
        let long_text = "This is a longer piece of text that contains multiple sentences. \
                        It tests whether the embedding model can handle longer inputs correctly. \
                        The embeddings should still be generated properly regardless of text length.";

        let request = create_text_request(vec![long_text.to_string()]);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle long text");

        let response = parse_response(result.unwrap()).expect("Should parse response");
        let embeddings = response.embeddings.expect("Should have embeddings");
        assert_eq!(response.dimension.unwrap_or(0), 384);
        assert!(embeddings[0].len() == 384);
    }

    #[tokio::test]
    async fn test_missing_input() {
        // Test when neither messages nor s3_file is provided
        let request = Request {
            messages: None,
            images: None,
            query: None,
            documents: None,
            s3_file: None,
            s3_images: None,
            save_to_s3: None,
            top_k: None,
            return_documents: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_err(),
            "Handler should fail when no input is provided"
        );

        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("Either 'messages' or 's3_file' must be provided"));
    }

    #[tokio::test]
    async fn test_empty_messages_array() {
        // Test with empty messages array
        let request = create_text_request(vec![]);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_err(),
            "Handler should fail with empty messages array"
        );
    }

    #[tokio::test]
    async fn test_embedding_consistency() {
        // Test that the same input produces consistent embeddings
        let text = "Consistent test message";

        let request1 = create_text_request(vec![text.to_string()]);
        let request2 = create_text_request(vec![text.to_string()]);

        let event1 = create_lambda_event(request1);
        let event2 = create_lambda_event(request2);

        let result1 = parse_response(handler(event1).await.unwrap()).unwrap();
        let result2 = parse_response(handler(event2).await.unwrap()).unwrap();

        let emb1 = result1.embeddings.unwrap();
        let emb2 = result2.embeddings.unwrap();

        assert_eq!(emb1[0].len(), emb2[0].len());

        // Check that embeddings are identical (or very close due to floating point)
        let embeddings_match = emb1[0]
            .iter()
            .zip(emb2[0].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);

        assert!(
            embeddings_match,
            "Same input should produce identical embeddings"
        );
    }

    #[tokio::test]
    async fn test_different_texts_produce_different_embeddings() {
        // Test that different texts produce different embeddings
        let request = create_text_request(vec![
            "Hello, world!".to_string(),
            "Goodbye, world!".to_string(),
        ]);

        let event = create_lambda_event(request);
        let result = parse_response(handler(event).await.unwrap()).unwrap();
        let embeddings = result.embeddings.unwrap();

        // Check that embeddings are different
        let embeddings_different = embeddings[0]
            .iter()
            .zip(embeddings[1].iter())
            .any(|(a, b)| (a - b).abs() > 0.01);

        assert!(
            embeddings_different,
            "Different texts should produce different embeddings"
        );
    }

    #[tokio::test]
    async fn test_embedding_vector_properties() {
        // Test mathematical properties of embedding vectors
        let request = create_text_request(vec!["Test vector properties".to_string()]);

        let event = create_lambda_event(request);
        let result = handler(event).await.unwrap();
        let response = parse_response(result).unwrap();

        let embeddings = response.embeddings.unwrap();
        let embedding = &embeddings[0];

        // Check all values are finite
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "All embedding values should be finite"
        );

        // Check vector has reasonable magnitude (not all zeros)
        let magnitude: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            magnitude > 0.0,
            "Embedding vector should have non-zero magnitude"
        );

        // BGE embeddings are typically normalized, so magnitude should be close to 1.0
        assert!(
            (magnitude - 1.0).abs() < 0.1,
            "BGE embeddings should be approximately normalized (magnitude ≈ 1.0)"
        );
    }

    #[tokio::test]
    async fn test_large_batch() {
        // Test with a larger batch of messages
        let messages: Vec<String> = (0..10).map(|i| format!("Message number {}", i)).collect();

        let request = create_text_request(messages);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle large batch");

        let response = parse_response(result.unwrap()).unwrap();
        let embeddings = response.embeddings.unwrap();
        assert_eq!(embeddings.len(), 10, "Should have 10 embeddings");

        for embedding in &embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }

    #[tokio::test]
    async fn test_special_characters() {
        // Test with special characters and Unicode
        let request =
            create_text_request(vec!["Hello 世界! 🌍 Special chars: @#$%^&*()".to_string()]);

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle special characters");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension.unwrap_or(0), 384);
    }

    #[tokio::test]
    async fn test_response_serialization() {
        // Test that the response can be serialized to JSON
        let request = create_text_request(vec!["Test serialization".to_string()]);

        let event = create_lambda_event(request);
        let result = handler(event).await.unwrap();
        let response = parse_response(result).unwrap();

        // Try to serialize the response
        let json = serde_json::to_string(&response);
        assert!(json.is_ok(), "Response should be serializable to JSON");

        let json_value = json.unwrap();
        assert!(json_value.contains("embeddings"));
        assert!(json_value.contains("model_type"));
    }

    #[tokio::test]
    async fn test_request_deserialization() {
        // Test that requests can be deserialized from JSON
        let json_str = r#"{
            "messages": ["Test message"]
        }"#;

        let request: Result<Request, _> = serde_json::from_str(json_str);
        assert!(request.is_ok(), "Request should deserialize from JSON");

        let req = request.unwrap();
        assert_eq!(req.messages.as_ref().unwrap()[0], "Test message");
    }

    #[tokio::test]
    async fn test_semantic_similarity() {
        // Test that semantically similar texts have similar embeddings
        let request = create_text_request(vec![
            "The cat sat on the mat".to_string(),
            "A feline rested on the rug".to_string(),
            "The weather is sunny today".to_string(),
        ]);

        let event = create_lambda_event(request);
        let result = parse_response(handler(event).await.unwrap()).unwrap();
        let embeddings = result.embeddings.unwrap();

        // Calculate cosine similarity
        let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (mag_a * mag_b)
        };

        let sim_similar = cosine_sim(&embeddings[0], &embeddings[1]);
        let sim_different = cosine_sim(&embeddings[0], &embeddings[2]);

        // Similar sentences should have higher similarity than different ones
        assert!(
            sim_similar > sim_different,
            "Semantically similar texts should have higher cosine similarity. Similar: {}, Different: {}",
            sim_similar,
            sim_different
        );
    }
}

#[cfg(test)]
mod save_config_tests {
    use super::*;

    #[test]
    fn test_save_config_deserialization() {
        let json_str = r#"{
            "bucket": "my-bucket",
            "key": "embeddings/test.json"
        }"#;

        let config: Result<SaveConfig, _> = serde_json::from_str(json_str);
        assert!(config.is_ok(), "SaveConfig should deserialize from JSON");

        let cfg = config.unwrap();
        assert_eq!(cfg.bucket, "my-bucket");
        assert_eq!(cfg.key, "embeddings/test.json");
    }
}

#[cfg(test)]
mod api_gateway_tests {
    use super::*;
    use serverless_vectorizer::ApiGatewayResponse;

    // Helper function to create an API Gateway event
    fn create_api_gateway_event(body: &str) -> LambdaEvent<Value> {
        let payload = json!({
            "httpMethod": "POST",
            "path": "/embed",
            "headers": {
                "Content-Type": "application/json"
            },
            "requestContext": {
                "requestId": "test-request-id",
                "accountId": "123456789012"
            },
            "body": body
        });

        LambdaEvent {
            payload,
            context: create_test_context(),
        }
    }

    // Helper function to parse API Gateway response
    fn parse_api_gateway_response(value: Value) -> Result<ApiGatewayResponse, String> {
        serde_json::from_value(value)
            .map_err(|e| format!("Failed to parse API Gateway response: {}", e))
    }

    #[tokio::test]
    async fn test_api_gateway_simple_message() {
        let body = r#"{"messages": ["Hello from API Gateway!"]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(
            result.is_ok(),
            "Handler should succeed for API Gateway request"
        );

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200, "Should return 200 status");
        assert_eq!(
            api_response.headers.get("Content-Type").unwrap(),
            "application/json"
        );

        // Parse the body to check the actual response
        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        assert_eq!(response.dimension.unwrap_or(0), 384);
        let embeddings = response.embeddings.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert!(embeddings[0].len() > 0);
        assert!(response.s3_location.is_none());
    }

    #[tokio::test]
    async fn test_api_gateway_multiple_messages() {
        let body = r#"{"messages": ["First message", "Second message", "Third message"]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(
            result.is_ok(),
            "Handler should succeed for multiple messages"
        );

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200);

        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        let embeddings = response.embeddings.unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(response.dimension.unwrap_or(0), 384);
    }

    #[tokio::test]
    async fn test_api_gateway_missing_body() {
        let payload = json!({
            "httpMethod": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "requestContext": {
                "requestId": "test-request-id"
            }
            // No body field
        });

        let event = LambdaEvent {
            payload,
            context: create_test_context(),
        };

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should return error response");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(
            api_response.status_code, 400,
            "Should return 400 for missing body"
        );

        let error_body: Value = serde_json::from_str(&api_response.body).unwrap();
        assert!(error_body.get("error").is_some());
        assert!(error_body["error"]
            .as_str()
            .unwrap()
            .contains("Missing body"));
    }

    #[tokio::test]
    async fn test_api_gateway_empty_messages_array() {
        let body = r#"{"messages": []}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should return error response");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 500);
    }
}

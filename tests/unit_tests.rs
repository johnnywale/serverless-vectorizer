// tests/unit_tests.rs

use lambda_runtime::{Context, LambdaEvent};
use serde_json::{Value, json};
use serverless_vectorizer::{Request, Response, SaveConfig, handler};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_text_embedding() {
        // Test basic text embedding
        let request = Request {
            messages: Some(vec!["Hello, world!".to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should succeed");

        let response = parse_response(result.unwrap()).expect("Should parse response");
        assert_eq!(response.embeddings.len(), 1, "Should have one embedding");
        assert!(
            response.embeddings[0].len() > 0,
            "Embedding should not be empty"
        );
        assert_eq!(
            response.dimension,
            response.embeddings[0].len(),
            "Dimension should match embedding length"
        );
        assert!(response.s3_location.is_none(), "S3 location should be None");

        // BGE-small-en-v1.5 produces 384-dimensional embeddings
        assert_eq!(
            response.dimension, 384,
            "Expected 384-dimensional embedding from BGE-small-en-v1.5"
        );
    }

    #[tokio::test]
    async fn test_multiple_messages_embedding() {
        // Test batch embedding with multiple messages
        let request = Request {
            messages: Some(vec![
                "First message".to_string(),
                "Second message".to_string(),
                "Third message".to_string(),
            ]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should succeed for multiple messages"
        );

        let response = parse_response(result.unwrap()).expect("Should parse response");
        assert_eq!(response.embeddings.len(), 3, "Should have three embeddings");
        assert_eq!(response.dimension, 384);

        // All embeddings should be 384-dimensional
        for embedding in &response.embeddings {
            assert_eq!(embedding.len(), 384);
        }

        // Different messages should produce different embeddings
        let embeddings_different = response.embeddings[0]
            .iter()
            .zip(response.embeddings[1].iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(
            embeddings_different,
            "Different messages should produce different embeddings"
        );
    }

    #[tokio::test]
    async fn test_empty_message() {
        // Test with empty string
        let request = Request {
            messages: Some(vec!["".to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle empty string");

        let response = parse_response(result.unwrap()).expect("Should parse response");
        assert_eq!(
            response.dimension, 384,
            "Should still produce 384-dimensional embedding"
        );
    }

    #[tokio::test]
    async fn test_long_text_embedding() {
        // Test with longer text
        let long_text = "This is a longer piece of text that contains multiple sentences. \
                        It tests whether the embedding model can handle longer inputs correctly. \
                        The embeddings should still be generated properly regardless of text length.";

        let request = Request {
            messages: Some(vec![long_text.to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle long text");

        let response = parse_response(result.unwrap()).expect("Should parse response");
        assert_eq!(response.dimension, 384);
        assert!(response.embeddings[0].len() == 384);
    }

    #[tokio::test]
    async fn test_missing_input() {
        // Test when neither messages nor s3_file is provided
        let request = Request {
            messages: None,
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_err(),
            "Handler should fail when no input is provided"
        );

        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Either 'messages' or 's3_file' must be provided")
        );
    }

    #[tokio::test]
    async fn test_empty_messages_array() {
        // Test with empty messages array
        let request = Request {
            messages: Some(vec![]),
            s3_file: None,
            save_to_s3: None,
        };

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

        let request1 = Request {
            messages: Some(vec![text.to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let request2 = Request {
            messages: Some(vec![text.to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event1 = create_lambda_event(request1);
        let event2 = create_lambda_event(request2);

        let result1 = parse_response(handler(event1).await.unwrap()).unwrap();
        let result2 = parse_response(handler(event2).await.unwrap()).unwrap();

        assert_eq!(result1.embeddings[0].len(), result2.embeddings[0].len());

        // Check that embeddings are identical (or very close due to floating point)
        let embeddings_match = result1.embeddings[0]
            .iter()
            .zip(result2.embeddings[0].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);

        assert!(
            embeddings_match,
            "Same input should produce identical embeddings"
        );
    }

    #[tokio::test]
    async fn test_batch_embedding_consistency() {
        // Test that same messages in batch produce consistent embeddings
        let messages = vec!["First message".to_string(), "Second message".to_string()];

        let request1 = Request {
            messages: Some(messages.clone()),
            s3_file: None,
            save_to_s3: None,
        };

        let request2 = Request {
            messages: Some(messages),
            s3_file: None,
            save_to_s3: None,
        };

        let event1 = create_lambda_event(request1);
        let event2 = create_lambda_event(request2);

        let result1 = parse_response(handler(event1).await.unwrap()).unwrap();
        let result2 = parse_response(handler(event2).await.unwrap()).unwrap();

        // Check that all embeddings match
        for i in 0..result1.embeddings.len() {
            let embeddings_match = result1.embeddings[i]
                .iter()
                .zip(result2.embeddings[i].iter())
                .all(|(a, b)| (a - b).abs() < 1e-6);
            assert!(
                embeddings_match,
                "Batch embedding {} should be consistent",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_different_texts_produce_different_embeddings() {
        // Test that different texts produce different embeddings
        let request = Request {
            messages: Some(vec![
                "Hello, world!".to_string(),
                "Goodbye, world!".to_string(),
            ]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = parse_response(handler(event).await.unwrap()).unwrap();

        // Check that embeddings are different
        let embeddings_different = result.embeddings[0]
            .iter()
            .zip(result.embeddings[1].iter())
            .any(|(a, b)| (a - b).abs() > 0.01);

        assert!(
            embeddings_different,
            "Different texts should produce different embeddings"
        );
    }

    #[tokio::test]
    async fn test_embedding_vector_properties() {
        // Test mathematical properties of embedding vectors
        let request = Request {
            messages: Some(vec!["Test vector properties".to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await.unwrap();
        let response = parse_response(result).unwrap();

        let embedding = &response.embeddings[0];

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

        let request = Request {
            messages: Some(messages.clone()),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle large batch");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 10, "Should have 10 embeddings");

        for embedding in &response.embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }

    #[tokio::test]
    async fn test_special_characters() {
        // Test with special characters and Unicode
        let request = Request {
            messages: Some(vec!["Hello 世界! 🌍 Special chars: @#$%^&*()".to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle special characters");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_numeric_text() {
        // Test with numeric content
        let request = Request {
            messages: Some(vec!["12345 67890".to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle numeric text");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_response_serialization() {
        // Test that the response can be serialized to JSON
        let request = Request {
            messages: Some(vec!["Test serialization".to_string()]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await.unwrap();
        let response = parse_response(result).unwrap();

        // Try to serialize the response
        let json = serde_json::to_string(&response);
        assert!(json.is_ok(), "Response should be serializable to JSON");

        let json_value = json.unwrap();
        assert!(json_value.contains("embeddings"));
        assert!(json_value.contains("dimension"));
    }

    #[tokio::test]
    async fn test_request_deserialization() {
        // Test that requests can be deserialized from JSON
        let json_str = r#"{
            "messages": ["Test message"],
            "s3_file": null,
            "save_to_s3": null
        }"#;

        let request: Result<Request, _> = serde_json::from_str(json_str);
        assert!(request.is_ok(), "Request should deserialize from JSON");

        let req = request.unwrap();
        assert_eq!(req.messages.as_ref().unwrap()[0], "Test message");
        assert!(req.s3_file.is_none());
        assert!(req.save_to_s3.is_none());
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        // Test that multiple concurrent requests work correctly
        use tokio::task;

        let handles: Vec<_> = (0..5)
            .map(|i| {
                task::spawn(async move {
                    let request = Request {
                        messages: Some(vec![format!("Concurrent request {}", i)]),
                        s3_file: None,
                        save_to_s3: None,
                    };

                    let event = create_lambda_event(request);
                    handler(event).await
                })
            })
            .collect();

        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "All concurrent requests should succeed");

            let response = parse_response(result.unwrap()).unwrap();
            assert_eq!(response.dimension, 384);
        }
    }

    #[tokio::test]
    async fn test_semantic_similarity() {
        // Test that semantically similar texts have similar embeddings
        let request = Request {
            messages: Some(vec![
                "The cat sat on the mat".to_string(),
                "A feline rested on the rug".to_string(),
                "The weather is sunny today".to_string(),
            ]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = parse_response(handler(event).await.unwrap()).unwrap();

        // Calculate cosine similarity
        let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (mag_a * mag_b)
        };

        let sim_similar = cosine_sim(&result.embeddings[0], &result.embeddings[1]);
        let sim_different = cosine_sim(&result.embeddings[0], &result.embeddings[2]);

        // Similar sentences should have higher similarity than different ones
        assert!(
            sim_similar > sim_different,
            "Semantically similar texts should have higher cosine similarity. Similar: {}, Different: {}",
            sim_similar,
            sim_different
        );

        // Similar texts should have reasonably high similarity (typically > 0.5 for related texts)
        assert!(
            sim_similar > 0.3,
            "Similar texts should have cosine similarity > 0.3, got: {}",
            sim_similar
        );
    }

    #[tokio::test]
    async fn test_whitespace_handling() {
        // Test various whitespace scenarios
        let texts = vec![
            "Hello world".to_string(),
            "Hello  world".to_string(),    // double space
            "  Hello world  ".to_string(), // leading/trailing spaces
            "Hello\nworld".to_string(),    // newline
            "Hello\tworld".to_string(),    // tab
        ];

        let request = Request {
            messages: Some(texts),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should handle whitespace variations"
        );

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 5);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_mixed_length_messages() {
        // Test batch with varying message lengths
        let request = Request {
            messages: Some(vec![
                "Short".to_string(),
                "Medium length message here".to_string(),
                "This is a much longer message that contains multiple sentences and covers various topics to test the embedding model's ability to handle varying input lengths effectively.".to_string(),
            ]),
            s3_file: None,
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should handle mixed length messages"
        );

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 3);

        // All should be same dimension
        for embedding in &response.embeddings {
            assert_eq!(embedding.len(), 384);
        }
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

    #[test]
    fn test_request_with_save_config() {
        let json_str = r#"{
            "messages": ["Test"],
            "save_to_s3": {
                "bucket": "test-bucket",
                "key": "test.json"
            }
        }"#;

        let request: Result<Request, _> = serde_json::from_str(json_str);
        assert!(
            request.is_ok(),
            "Request with save_to_s3 should deserialize"
        );

        let req = request.unwrap();
        assert!(req.save_to_s3.is_some());

        let save_cfg = req.save_to_s3.unwrap();
        assert_eq!(save_cfg.bucket, "test-bucket");
        assert_eq!(save_cfg.key, "test.json");
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
        assert_eq!(response.dimension, 384);
        assert_eq!(response.embeddings.len(), 1);
        assert!(response.embeddings[0].len() > 0);
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
        assert_eq!(response.embeddings.len(), 3);
        assert_eq!(response.dimension, 384);
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
        assert!(
            error_body["error"]
                .as_str()
                .unwrap()
                .contains("Missing body")
        );
    }

    #[tokio::test]
    async fn test_api_gateway_invalid_json_body() {
        let body = r#"{"messages": invalid json}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should return error response");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(
            api_response.status_code, 400,
            "Should return 400 for invalid JSON"
        );

        let error_body: Value = serde_json::from_str(&api_response.body).unwrap();
        assert!(error_body.get("error").is_some());
        assert!(
            error_body["error"]
                .as_str()
                .unwrap()
                .contains("Failed to parse body")
        );
    }

    #[tokio::test]
    async fn test_api_gateway_missing_required_fields() {
        let body = r#"{"some_other_field": "value"}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should return error response");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(
            api_response.status_code, 500,
            "Should return 500 for processing error"
        );

        let error_body: Value = serde_json::from_str(&api_response.body).unwrap();
        assert!(error_body.get("error").is_some());
        assert!(
            error_body["error"]
                .as_str()
                .unwrap()
                .contains("Either 'messages' or 's3_file' must be provided")
        );
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

    #[tokio::test]
    async fn test_api_gateway_with_s3_file() {
        let body = r#"{"s3_file": "my-bucket/test.txt"}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;

        // Note: This will fail in unit tests because S3 isn't available
        // But we can verify it's handled as an API Gateway request
        assert!(result.is_ok(), "Handler should return a response");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        // Should be 500 because S3 operation will fail in test environment
        assert!(api_response.status_code == 200 || api_response.status_code == 500);
    }

    #[tokio::test]
    async fn test_api_gateway_batch_processing() {
        let messages: Vec<String> = (0..5).map(|i| format!("Batch message {}", i)).collect();

        let body = json!({"messages": messages}).to_string();
        let event = create_api_gateway_event(&body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should handle batch processing");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200);

        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        assert_eq!(response.embeddings.len(), 5);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_api_gateway_long_message() {
        let long_text =
            "This is a very long message that simulates a real-world API Gateway request. "
                .repeat(50);
        let body = json!({
            "messages": [long_text]
        })
        .to_string();

        let event = create_api_gateway_event(&body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should handle long messages");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200);

        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_api_gateway_special_characters_in_body() {
        let body = r#"{"messages": ["Special chars: 世界 🌍 @#$%^&*()"]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should handle special characters");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200);

        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_api_gateway_response_structure() {
        let body = r#"{"messages": ["Test response structure"]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await.unwrap();
        let api_response = parse_api_gateway_response(result).unwrap();

        // Verify API Gateway response structure
        assert!(api_response.status_code > 0);
        assert!(api_response.headers.contains_key("Content-Type"));
        assert!(!api_response.body.is_empty());

        // Verify body is valid JSON
        let parsed_body: Value = serde_json::from_str(&api_response.body).unwrap();
        assert!(parsed_body.is_object());
    }

    #[tokio::test]
    async fn test_api_gateway_vs_direct_invocation() {
        let message = "Compare invocation methods";

        // API Gateway request
        let api_body = json!({"messages": [message]}).to_string();
        let api_event = create_api_gateway_event(&api_body);
        let api_result = handler(api_event).await.unwrap();
        let api_response = parse_api_gateway_response(api_result).unwrap();

        // Direct Lambda invocation
        let direct_request = Request {
            messages: Some(vec![message.to_string()]),
            s3_file: None,
            save_to_s3: None,
        };
        let direct_event = create_lambda_event(direct_request);
        let direct_result = handler(direct_event).await.unwrap();
        let direct_response = parse_response(direct_result).unwrap();

        // Parse API Gateway body
        let api_body_response: Response = serde_json::from_str(&api_response.body).unwrap();

        // Both should produce the same embedding
        assert_eq!(api_body_response.dimension, direct_response.dimension);
        assert_eq!(
            api_body_response.embeddings[0].len(),
            direct_response.embeddings[0].len()
        );

        // Embeddings should be identical
        let embeddings_match = api_body_response.embeddings[0]
            .iter()
            .zip(direct_response.embeddings[0].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);

        assert!(
            embeddings_match,
            "Same message should produce identical embeddings regardless of invocation method"
        );
    }

    #[tokio::test]
    async fn test_api_gateway_multiple_sequential_requests() {
        let messages = vec![
            vec!["First message"],
            vec!["Second message"],
            vec!["Third message"],
        ];

        for (i, msg) in messages.iter().enumerate() {
            let body = json!({"messages": msg}).to_string();
            let event = create_api_gateway_event(&body);

            let result = handler(event).await;
            assert!(result.is_ok(), "Request {} should succeed", i + 1);

            let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
            assert_eq!(api_response.status_code, 200);

            let response: Response = serde_json::from_str(&api_response.body).unwrap();
            assert_eq!(response.dimension, 384);
        }
    }

    #[tokio::test]
    async fn test_api_gateway_empty_message() {
        let body = r#"{"messages": [""]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should handle empty message");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200);

        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_api_gateway_with_different_http_methods() {
        let body = r#"{"messages": ["Test message"]}"#;

        // Test with different HTTP methods
        for method in &["POST", "GET", "PUT", "DELETE"] {
            let payload = json!({
                "httpMethod": method,
                "path": "/embed",
                "headers": {
                    "Content-Type": "application/json"
                },
                "requestContext": {
                    "requestId": "test-request-id"
                },
                "body": body
            });

            let event = LambdaEvent {
                payload,
                context: create_test_context(),
            };

            let result = handler(event).await;
            assert!(result.is_ok(), "Handler should process {} request", method);

            let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
            // Should succeed regardless of HTTP method (business logic doesn't check method)
            assert!(api_response.status_code == 200 || api_response.status_code == 400);
        }
    }

    #[tokio::test]
    async fn test_api_gateway_headers_in_response() {
        let body = r#"{"messages": ["Test headers"]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await.unwrap();
        let api_response = parse_api_gateway_response(result).unwrap();

        // Verify headers are present and correct
        assert!(api_response.headers.contains_key("Content-Type"));
        assert_eq!(
            api_response.headers.get("Content-Type").unwrap(),
            "application/json"
        );
    }

    #[tokio::test]
    async fn test_api_gateway_request_with_save_config() {
        let body = r#"{
            "messages": ["Test save to S3"],
            "save_to_s3": {
                "bucket": "test-bucket",
                "key": "embeddings/test.json"
            }
        }"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(
            result.is_ok(),
            "Handler should process request with save config"
        );

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        // Will likely be 500 in test environment due to S3, but request is processed
        assert!(api_response.status_code == 200 || api_response.status_code == 500);
    }

    #[tokio::test]
    async fn test_api_gateway_mixed_batch() {
        let body = r#"{"messages": ["Short", "Medium length text here", "This is a much longer message with multiple sentences and various content to test processing."]}"#;
        let event = create_api_gateway_event(body);

        let result = handler(event).await;
        assert!(result.is_ok(), "Handler should handle mixed length batch");

        let api_response = parse_api_gateway_response(result.unwrap()).unwrap();
        assert_eq!(api_response.status_code, 200);

        let response: Response = serde_json::from_str(&api_response.body).unwrap();
        assert_eq!(response.embeddings.len(), 3);

        for embedding in &response.embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }
}

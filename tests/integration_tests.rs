// tests/integration_tests.rs

use aws_config::{BehaviorVersion, Region};
use aws_sdk_s3 as s3;
use aws_sdk_s3::config::Credentials;
use serverless_vectorizer::{Request, Response, SaveConfig, handler};
use lambda_runtime::{Context, LambdaEvent};
use serde_json::Value;

// Setup LocalStack environment variables so handler's aws_config::load_from_env() uses LocalStack
fn setup_localstack_env() {
    let endpoint_url = std::env::var("LOCALSTACK_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:4566".to_string());

    // SAFETY: These tests run sequentially with --test-threads=1 or the env vars
    // are set before any concurrent access. This is safe for test setup.
    unsafe {
        std::env::set_var("AWS_ENDPOINT_URL", &endpoint_url);
        std::env::set_var("AWS_ACCESS_KEY_ID", "test");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test");
        std::env::set_var("AWS_REGION", "us-east-1");
    }
}

// Helper function to create S3 client for LocalStack
async fn create_localstack_s3_client() -> s3::Client {
    let endpoint_url = std::env::var("LOCALSTACK_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:4566".to_string());

    let creds = Credentials::new("test", "test", None, None, "test");

    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(creds)
        .endpoint_url(&endpoint_url)
        .load()
        .await;

    // Create S3-specific config with force_path_style enabled
    let s3_config = s3::config::Builder::from(&config)
        .force_path_style(true)  // Add this line!
        .build();

    s3::Client::from_conf(s3_config)
}

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

// Helper function to create bucket if it doesn't exist
async fn ensure_bucket_exists(
    s3_client: &s3::Client,
    bucket: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match s3_client.head_bucket().bucket(bucket).send().await {
        Ok(_) => Ok(()),
        Err(_) => {
            s3_client.create_bucket().bucket(bucket).send().await?;
            Ok(())
        }
    }
}

// Helper function to upload text to S3
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

// Helper function to upload JSON array to S3
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

// Helper function to read from S3
async fn read_from_s3(
    s3_client: &s3::Client,
    bucket: &str,
    key: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let resp = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;

    let data = resp.body.collect().await?;
    Ok(String::from_utf8(data.to_vec())?)
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    const TEST_BUCKET: &str = "test-embeddings-bucket";

    #[tokio::test]
    async fn test_read_single_text_from_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let test_text = "Hello from S3!";
        let s3_key = "test-files/single-text.txt";

        upload_text_to_s3(&s3_client, TEST_BUCKET, s3_key, test_text)
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, s3_key)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should succeed reading from S3");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_read_json_array_from_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
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

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, s3_key)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should succeed reading JSON array from S3"
        );

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 3, "Should generate 3 embeddings");
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_save_embeddings_to_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let output_key = "test-output/embeddings.json";

        // Test
        let request = Request {
            messages: Some(vec!["Test message for saving".to_string()]),
            s3_file: None,
            save_to_s3: Some(SaveConfig {
                bucket: TEST_BUCKET.to_string(),
                key: output_key.to_string(),
            }),
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should succeed saving to S3");

        let response = parse_response(result.unwrap()).unwrap();
        assert!(response.s3_location.is_some());
        assert_eq!(
            response.s3_location.unwrap(),
            format!("s3://{}/{}", TEST_BUCKET, output_key)
        );

        // Verify the file was actually saved
        let saved_content = read_from_s3(&s3_client, TEST_BUCKET, output_key)
            .await
            .unwrap();

        let saved_embeddings: Vec<Vec<f32>> = serde_json::from_str(&saved_content).unwrap();
        assert_eq!(saved_embeddings.len(), 1);
        assert_eq!(saved_embeddings[0].len(), 384);
    }

    #[tokio::test]
    async fn test_save_multiple_embeddings_to_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let output_key = "test-output/batch-embeddings.json";

        // Test
        let request = Request {
            messages: Some(vec![
                "First message".to_string(),
                "Second message".to_string(),
                "Third message".to_string(),
            ]),
            s3_file: None,
            save_to_s3: Some(SaveConfig {
                bucket: TEST_BUCKET.to_string(),
                key: output_key.to_string(),
            }),
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should succeed saving batch to S3");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 3);

        // Verify the file was actually saved with all embeddings
        let saved_content = read_from_s3(&s3_client, TEST_BUCKET, output_key)
            .await
            .unwrap();

        let saved_embeddings: Vec<Vec<f32>> = serde_json::from_str(&saved_content).unwrap();
        assert_eq!(saved_embeddings.len(), 3);

        for embedding in &saved_embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }

    #[tokio::test]
    async fn test_read_from_s3_and_save_to_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let input_key = "test-files/input-text.txt";
        let output_key = "test-output/result-embeddings.json";
        let test_text = "Process this text from S3 and save result";

        upload_text_to_s3(&s3_client, TEST_BUCKET, input_key, test_text)
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, input_key)),
            save_to_s3: Some(SaveConfig {
                bucket: TEST_BUCKET.to_string(),
                key: output_key.to_string(),
            }),
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should succeed with both S3 read and write"
        );

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert!(response.s3_location.is_some());

        // Verify output file
        let saved_content = read_from_s3(&s3_client, TEST_BUCKET, output_key)
            .await
            .unwrap();

        let saved_embeddings: Vec<Vec<f32>> = serde_json::from_str(&saved_content).unwrap();
        assert_eq!(saved_embeddings.len(), 1);
        assert_eq!(saved_embeddings[0], response.embeddings[0]);
    }

    #[tokio::test]
    async fn test_read_json_array_and_save_to_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let input_key = "test-files/batch-input.json";
        let output_key = "test-output/batch-output.json";
        let messages = vec!["Batch message 1".to_string(), "Batch message 2".to_string()];

        upload_json_to_s3(&s3_client, TEST_BUCKET, input_key, &messages)
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, input_key)),
            save_to_s3: Some(SaveConfig {
                bucket: TEST_BUCKET.to_string(),
                key: output_key.to_string(),
            }),
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_ok(),
            "Handler should succeed with batch S3 read and write"
        );

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.embeddings.len(), 2);

        // Verify output
        let saved_content = read_from_s3(&s3_client, TEST_BUCKET, output_key)
            .await
            .unwrap();

        let saved_embeddings: Vec<Vec<f32>> = serde_json::from_str(&saved_content).unwrap();
        assert_eq!(saved_embeddings.len(), 2);
    }

    #[tokio::test]
    async fn test_invalid_s3_path() {
        setup_localstack_env();
        let request = Request {
            messages: None,
            s3_file: Some("invalid-path".to_string()),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_err(), "Handler should fail with invalid S3 path");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid S3 path format")
        );
    }

    #[tokio::test]
    async fn test_nonexistent_s3_file() {
        setup_localstack_env();
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/nonexistent-file.txt", TEST_BUCKET)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(
            result.is_err(),
            "Handler should fail with nonexistent S3 file"
        );
    }

    #[tokio::test]
    async fn test_large_file_from_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let large_text = "This is a test sentence. ".repeat(100);
        let s3_key = "test-files/large-file.txt";

        upload_text_to_s3(&s3_client, TEST_BUCKET, s3_key, &large_text)
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, s3_key)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle large files");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_unicode_content_from_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let unicode_text = "Hello 世界! 🌍 Привет مرحبا";
        let s3_key = "test-files/unicode.txt";

        upload_text_to_s3(&s3_client, TEST_BUCKET, s3_key, unicode_text)
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, s3_key)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle Unicode content");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_empty_file_from_s3() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let s3_key = "test-files/empty.txt";

        upload_text_to_s3(&s3_client, TEST_BUCKET, s3_key, "")
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, s3_key)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle empty files");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_multiple_files_sequential() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let files = vec![
            ("test-files/seq-1.txt", "First sequential file"),
            ("test-files/seq-2.txt", "Second sequential file"),
            ("test-files/seq-3.txt", "Third sequential file"),
        ];

        for (key, content) in &files {
            upload_text_to_s3(&s3_client, TEST_BUCKET, key, content)
                .await
                .unwrap();
        }

        // Test each file
        for (key, _) in files {
            let request = Request {
                messages: None,
                s3_file: Some(format!("{}/{}", TEST_BUCKET, key)),
                save_to_s3: None,
            };

            let event = create_lambda_event(request);
            let result = handler(event).await;

            assert!(result.is_ok(), "Handler should succeed for file: {}", key);

            let response = parse_response(result.unwrap()).unwrap();
            assert_eq!(response.dimension, 384);
        }
    }

    #[tokio::test]
    async fn test_nested_s3_paths() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let nested_key = "level1/level2/level3/nested-file.txt";
        let test_text = "Content in nested path";

        upload_text_to_s3(&s3_client, TEST_BUCKET, nested_key, test_text)
            .await
            .unwrap();

        // Test
        let request = Request {
            messages: None,
            s3_file: Some(format!("{}/{}", TEST_BUCKET, nested_key)),
            save_to_s3: None,
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        assert!(result.is_ok(), "Handler should handle nested S3 paths");

        let response = parse_response(result.unwrap()).unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_concurrent_s3_operations() {
        use tokio::task;

        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        // Upload test files
        for i in 0..5 {
            let key = format!("test-files/concurrent-{}.txt", i);
            let content = format!("Concurrent test message {}", i);
            upload_text_to_s3(&s3_client, TEST_BUCKET, &key, &content)
                .await
                .unwrap();
        }

        // Test concurrent processing
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let bucket = TEST_BUCKET.to_string();
                task::spawn(async move {
                    let request = Request {
                        messages: None,
                        s3_file: Some(format!("{}/test-files/concurrent-{}.txt", bucket, i)),
                        save_to_s3: None,
                    };

                    let event = create_lambda_event(request);
                    handler(event).await
                })
            })
            .collect();

        // Verify all succeed
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(
                result.is_ok(),
                "All concurrent S3 operations should succeed"
            );

            let response = parse_response(result.unwrap()).unwrap();
            assert_eq!(response.dimension, 384);
        }
    }

    #[tokio::test]
    async fn test_save_to_nonexistent_bucket() {
        setup_localstack_env();
        let request = Request {
            messages: Some(vec!["Test message".to_string()]),
            s3_file: None,
            save_to_s3: Some(SaveConfig {
                bucket: "nonexistent-bucket-12345".to_string(),
                key: "test.json".to_string(),
            }),
        };

        let event = create_lambda_event(request);
        let result = handler(event).await;

        // Should fail because bucket doesn't exist
        assert!(
            result.is_err(),
            "Handler should fail with nonexistent bucket"
        );
    }

    #[tokio::test]
    async fn test_overwrite_existing_s3_file() {
        // Setup LocalStack env for handler
        setup_localstack_env();
        let s3_client = create_localstack_s3_client().await;
        ensure_bucket_exists(&s3_client, TEST_BUCKET).await.unwrap();

        let output_key = "test-output/overwrite-test.json";

        // First save
        let request1 = Request {
            messages: Some(vec!["First version".to_string()]),
            s3_file: None,
            save_to_s3: Some(SaveConfig {
                bucket: TEST_BUCKET.to_string(),
                key: output_key.to_string(),
            }),
        };

        let event1 = create_lambda_event(request1);
        handler(event1).await.unwrap();

        // Second save (overwrite)
        let request2 = Request {
            messages: Some(vec!["Second version".to_string()]),
            s3_file: None,
            save_to_s3: Some(SaveConfig {
                bucket: TEST_BUCKET.to_string(),
                key: output_key.to_string(),
            }),
        };

        let event2 = create_lambda_event(request2);
        let result = handler(event2).await;

        assert!(result.is_ok(), "Handler should succeed overwriting file");

        // Verify the file was overwritten
        let saved_content = read_from_s3(&s3_client, TEST_BUCKET, output_key)
            .await
            .unwrap();

        let saved_embeddings: Vec<Vec<f32>> = serde_json::from_str(&saved_content).unwrap();
        assert_eq!(saved_embeddings.len(), 1);
    }
}

// Image loading utilities for image embedding support

use crate::core::types::ImageInput;
use base64::Engine;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageError {
    #[error("Failed to decode base64 image: {0}")]
    Base64DecodeError(String),

    #[error("Failed to load image from file: {0}")]
    FileLoadError(String),

    #[error("Failed to load image from S3: {0}")]
    S3LoadError(String),

    #[error("Invalid image format: {0}")]
    InvalidFormat(String),

    #[error("Image input is empty or invalid")]
    EmptyInput,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Load image bytes from various input sources
pub fn load_image_bytes(input: &ImageInput) -> Result<Vec<u8>, ImageError> {
    match input {
        ImageInput::Base64 { base64 } => decode_base64_image(base64),
        ImageInput::FilePath { path } => load_image_from_file(path),
        ImageInput::S3Path { .. } => {
            // S3 loading requires async, so we return an error here
            // The async version should be used for S3
            Err(ImageError::S3LoadError(
                "Use load_image_bytes_async for S3 paths".to_string(),
            ))
        }
    }
}

/// Decode base64 encoded image data
pub fn decode_base64_image(data: &str) -> Result<Vec<u8>, ImageError> {
    // Handle data URL format (e.g., "data:image/png;base64,...")
    let base64_data = if data.contains(",") {
        data.split(",").last().unwrap_or(data)
    } else {
        data
    };

    // Remove whitespace
    let cleaned: String = base64_data.chars().filter(|c| !c.is_whitespace()).collect();

    if cleaned.is_empty() {
        return Err(ImageError::EmptyInput);
    }

    base64::engine::general_purpose::STANDARD
        .decode(&cleaned)
        .map_err(|e| ImageError::Base64DecodeError(e.to_string()))
}

/// Load image from local file path
pub fn load_image_from_file(path: &str) -> Result<Vec<u8>, ImageError> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(ImageError::FileLoadError(format!(
            "File not found: {}",
            path.display()
        )));
    }

    std::fs::read(path).map_err(|e| ImageError::FileLoadError(e.to_string()))
}

/// Load multiple images from various input sources (sync, no S3)
pub fn load_images_bytes(inputs: &[ImageInput]) -> Result<Vec<Vec<u8>>, ImageError> {
    inputs.iter().map(load_image_bytes).collect()
}

// ============================================================================
// Async S3 support (requires aws feature)
// ============================================================================

#[cfg(feature = "aws")]
pub mod s3 {
    use super::*;
    use aws_sdk_s3::Client as S3Client;

    /// Load image bytes from S3
    pub async fn load_image_from_s3(
        client: &S3Client,
        bucket: &str,
        key: &str,
    ) -> Result<Vec<u8>, ImageError> {
        let response = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| ImageError::S3LoadError(e.to_string()))?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| ImageError::S3LoadError(e.to_string()))?
            .into_bytes()
            .to_vec();

        Ok(bytes)
    }

    /// Parse S3 path in format "bucket/key" or "s3://bucket/key"
    pub fn parse_s3_path(path: &str) -> Result<(String, String), ImageError> {
        let path = path.strip_prefix("s3://").unwrap_or(path);

        let parts: Vec<&str> = path.splitn(2, '/').collect();
        if parts.len() != 2 {
            return Err(ImageError::S3LoadError(format!(
                "Invalid S3 path format: {}. Expected: bucket/key",
                path
            )));
        }

        Ok((parts[0].to_string(), parts[1].to_string()))
    }

    /// Load image bytes from ImageInput (async, supports S3)
    pub async fn load_image_bytes_async(
        input: &ImageInput,
        s3_client: Option<&S3Client>,
    ) -> Result<Vec<u8>, ImageError> {
        match input {
            ImageInput::Base64 { base64 } => decode_base64_image(base64),
            ImageInput::FilePath { path } => load_image_from_file(path),
            ImageInput::S3Path { s3_path } => {
                let client = s3_client.ok_or_else(|| {
                    ImageError::S3LoadError("S3 client not provided".to_string())
                })?;
                let (bucket, key) = parse_s3_path(s3_path)?;
                load_image_from_s3(client, &bucket, &key).await
            }
        }
    }

    /// Load multiple images (async, supports S3)
    pub async fn load_images_bytes_async(
        inputs: &[ImageInput],
        s3_client: Option<&S3Client>,
    ) -> Result<Vec<Vec<u8>>, ImageError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(load_image_bytes_async(input, s3_client).await?);
        }
        Ok(results)
    }
}

// ============================================================================
// Image validation utilities
// ============================================================================

/// Check if bytes look like a valid image based on magic bytes
pub fn is_valid_image_bytes(bytes: &[u8]) -> bool {
    if bytes.len() < 8 {
        return false;
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return true;
    }

    // JPEG: FF D8 FF
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return true;
    }

    // GIF: GIF87a or GIF89a
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return true;
    }

    // WebP: RIFF....WEBP
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return true;
    }

    // BMP: BM
    if bytes.starts_with(b"BM") {
        return true;
    }

    false
}

/// Get image format from magic bytes
pub fn detect_image_format(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() < 8 {
        return None;
    }

    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("gif");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("webp");
    }
    if bytes.starts_with(b"BM") {
        return Some("bmp");
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_base64_simple() {
        // A minimal valid PNG (1x1 transparent pixel)
        let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = decode_base64_image(png_base64);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(is_valid_image_bytes(&bytes));
        assert_eq!(detect_image_format(&bytes), Some("png"));
    }

    #[test]
    fn test_decode_base64_data_url() {
        let data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = decode_base64_image(data_url);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_base64() {
        let result = decode_base64_image("");
        assert!(matches!(result, Err(ImageError::EmptyInput)));
    }

    #[test]
    fn test_invalid_base64() {
        let result = decode_base64_image("not-valid-base64!!!");
        assert!(matches!(result, Err(ImageError::Base64DecodeError(_))));
    }

    #[test]
    fn test_image_input_constructors() {
        let base64 = ImageInput::from_base64("test".to_string());
        assert!(matches!(base64, ImageInput::Base64 { .. }));

        let path = ImageInput::from_path("/path/to/image.png");
        assert!(matches!(path, ImageInput::FilePath { .. }));

        let s3 = ImageInput::from_s3("bucket/key");
        assert!(matches!(s3, ImageInput::S3Path { .. }));
    }

    #[cfg(feature = "aws")]
    #[test]
    fn test_parse_s3_path() {
        use s3::parse_s3_path;

        let (bucket, key) = parse_s3_path("my-bucket/path/to/image.png").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/image.png");

        let (bucket, key) = parse_s3_path("s3://my-bucket/image.jpg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "image.jpg");
    }
}

// Embedding generation service

use crate::core::model::ModelType;
use fastembed::{InitOptions, TextEmbedding};
use std::collections::HashMap;
use std::sync::Mutex;
use thiserror::Error;

/// Errors that can occur during embedding operations
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Failed to initialize model: {0}")]
    ModelInitError(String),

    #[error("Failed to generate embeddings: {0}")]
    EmbeddingFailed(String),

    #[error("Model lock error: {0}")]
    LockError(String),

    #[error("Empty input: no texts provided")]
    EmptyInput,

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Thread-safe embedding service with model caching
pub struct EmbeddingService {
    models: Mutex<HashMap<ModelType, TextEmbedding>>,
    show_progress: bool,
}

impl EmbeddingService {
    /// Create a new embedding service
    pub fn new() -> Self {
        EmbeddingService {
            models: Mutex::new(HashMap::new()),
            show_progress: false,
        }
    }

    /// Create with download progress enabled
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Load a model if not already cached
    fn ensure_model(&self, model_type: ModelType) -> Result<(), EmbeddingError> {
        let mut models = self
            .models
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        if !models.contains_key(&model_type) {
            let model = TextEmbedding::try_new(
                InitOptions::new(model_type.to_fastembed())
                    .with_show_download_progress(self.show_progress),
            )
            .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

            models.insert(model_type, model);
        }

        Ok(())
    }

    /// Generate embeddings for a list of texts
    pub fn embed(
        &self,
        texts: Vec<String>,
        model_type: ModelType,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        self.ensure_model(model_type)?;

        let mut models = self
            .models
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let model = models
            .get_mut(&model_type)
            .ok_or_else(|| EmbeddingError::ModelInitError("Model not found in cache".to_string()))?;

        model
            .embed(texts, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))
    }

    /// Generate embedding for a single text
    pub fn embed_one(
        &self,
        text: &str,
        model_type: ModelType,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let embeddings = self.embed(vec![text.to_string()], model_type)?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::EmbeddingFailed("No embedding generated".to_string()))
    }

    /// Check if a model is loaded
    pub fn is_model_loaded(&self, model_type: ModelType) -> bool {
        self.models
            .lock()
            .map(|models| models.contains_key(&model_type))
            .unwrap_or(false)
    }

    /// Preload a model
    pub fn preload(&self, model_type: ModelType) -> Result<(), EmbeddingError> {
        self.ensure_model(model_type)
    }

    /// Get the dimension of embeddings for a model type
    pub fn dimension(&self, model_type: ModelType) -> usize {
        model_type.dimension()
    }
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

// Global embedding service instance for convenience
static GLOBAL_SERVICE: std::sync::LazyLock<EmbeddingService> =
    std::sync::LazyLock::new(|| EmbeddingService::new());

/// Get the global embedding service instance
pub fn global_service() -> &'static EmbeddingService {
    &GLOBAL_SERVICE
}

/// Convenience function to generate embeddings using global service
pub fn embed(texts: Vec<String>, model_type: ModelType) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    global_service().embed(texts, model_type)
}

/// Convenience function to generate single embedding using global service
pub fn embed_one(text: &str, model_type: ModelType) -> Result<Vec<f32>, EmbeddingError> {
    global_service().embed_one(text, model_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_service_creation() {
        let service = EmbeddingService::new();
        assert!(!service.is_model_loaded(ModelType::BgeSmallEnV15));
    }

    #[test]
    fn test_empty_input_error() {
        let service = EmbeddingService::new();
        let result = service.embed(vec![], ModelType::BgeSmallEnV15);
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }
}

// Embedding generation service supporting multiple model types

use crate::core::model::{ModelCategory, ModelRegistry, ModelType};
use crate::core::types::{RerankResult, SparseEmbedding};
use fastembed::{
    ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions,
    RerankerModel, SparseInitOptions, SparseModel, SparseTextEmbedding, TextEmbedding, TextRerank,
};
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

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Wrong model type: expected {expected}, got {actual}")]
    WrongModelType { expected: String, actual: String },

    #[error("Image loading error: {0}")]
    ImageLoadError(String),
}

// ============================================================================
// Text Embedding Service (existing, refactored)
// ============================================================================

/// Thread-safe text embedding service with model caching
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

    /// Unload a specific model from cache
    pub fn unload(&self, model_type: ModelType) -> Result<bool, EmbeddingError> {
        let mut models = self
            .models
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        Ok(models.remove(&model_type).is_some())
    }

    /// Unload all models from cache
    pub fn unload_all(&self) -> Result<usize, EmbeddingError> {
        let mut models = self
            .models
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let count = models.len();
        models.clear();
        Ok(count)
    }

    /// Reload a model (unload then preload)
    pub fn reload(&self, model_type: ModelType) -> Result<(), EmbeddingError> {
        self.unload(model_type)?;
        self.preload(model_type)
    }

    /// Get list of currently loaded models
    pub fn loaded_models(&self) -> Vec<ModelType> {
        self.models
            .lock()
            .map(|models| models.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Generate embeddings using a fastembed model directly
    pub fn embed_with_model(
        &self,
        texts: Vec<String>,
        model: fastembed::EmbeddingModel,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Create a new TextEmbedding for this model
        let mut embedding = TextEmbedding::try_new(
            InitOptions::new(model).with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        embedding
            .embed(texts, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))
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

// ============================================================================
// Image Embedding Service
// ============================================================================

/// Thread-safe image embedding service
pub struct ImageEmbeddingService {
    model: Mutex<Option<(ImageEmbeddingModel, ImageEmbedding)>>,
    show_progress: bool,
}

impl ImageEmbeddingService {
    pub fn new() -> Self {
        ImageEmbeddingService {
            model: Mutex::new(None),
            show_progress: false,
        }
    }

    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Initialize with a specific model
    pub fn init(&self, model_id: &str) -> Result<(), EmbeddingError> {
        let embedding_model = ModelRegistry::find_image_model(model_id)
            .ok_or_else(|| EmbeddingError::ModelNotFound(model_id.to_string()))?;

        let model = ImageEmbedding::try_new(
            ImageInitOptions::new(embedding_model.clone())
                .with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;
        *guard = Some((embedding_model, model));

        Ok(())
    }

    /// Generate embeddings for images (as byte arrays)
    /// Note: fastembed requires file paths, so we write temp files
    pub fn embed_images(&self, images: Vec<Vec<u8>>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if images.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Write images to temp files since fastembed expects paths
        // Use unique ID to avoid collisions when running in parallel
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut temp_paths: Vec<std::path::PathBuf> = Vec::with_capacity(images.len());

        for (i, image_bytes) in images.iter().enumerate() {
            // Detect image format and use appropriate extension
            let ext = crate::core::image_utils::detect_image_format(image_bytes).unwrap_or("png");
            let temp_path = temp_dir.join(format!("fastembed_img_{}_{}.{}", unique_id, i, ext));
            std::fs::write(&temp_path, image_bytes)
                .map_err(|e| EmbeddingError::ImageLoadError(e.to_string()))?;
            temp_paths.push(temp_path);
        }

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let (_, model) = guard
            .as_mut()
            .ok_or_else(|| EmbeddingError::ModelInitError("Model not initialized".to_string()))?;

        let result = model
            .embed(temp_paths.clone(), None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()));

        // Clean up temp files
        for path in temp_paths {
            let _ = std::fs::remove_file(path);
        }

        result
    }

    /// Generate embeddings for images from file paths
    pub fn embed_from_paths<P: AsRef<std::path::Path> + Send + Sync>(
        &self,
        paths: Vec<P>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if paths.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let (_, model) = guard
            .as_mut()
            .ok_or_else(|| EmbeddingError::ModelInitError("Model not initialized".to_string()))?;

        model
            .embed(paths, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))
    }

    /// Generate embedding for a single image
    pub fn embed_one(&self, image: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
        let embeddings = self.embed_images(vec![image.to_vec()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::EmbeddingFailed("No embedding generated".to_string()))
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get dimension of the loaded model
    pub fn dimension(&self) -> Option<usize> {
        self.model
            .lock()
            .ok()
            .and_then(|guard| {
                guard.as_ref().map(|(model_enum, _)| {
                    ImageEmbedding::list_supported_models()
                        .into_iter()
                        .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(model_enum))
                        .map(|info| info.dim)
                        .unwrap_or(512)
                })
            })
    }

    /// Unload the current model
    pub fn unload(&self) -> Result<bool, EmbeddingError> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let was_loaded = guard.is_some();
        *guard = None;
        Ok(was_loaded)
    }

    /// Reload the model with a new model ID
    pub fn reload(&self, model_id: &str) -> Result<(), EmbeddingError> {
        self.unload()?;
        self.init(model_id)
    }

    /// Get the currently loaded model ID
    pub fn loaded_model_id(&self) -> Option<String> {
        self.model
            .lock()
            .ok()
            .and_then(|guard| {
                guard.as_ref().map(|(model_enum, _)| {
                    ImageEmbedding::list_supported_models()
                        .into_iter()
                        .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(model_enum))
                        .map(|info| info.model_code.to_string())
                        .unwrap_or_default()
                })
            })
    }

    /// Generate embeddings for images using a specific model
    pub fn embed_images_with_model(
        &self,
        images: &[Vec<u8>],
        model: ImageEmbeddingModel,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if images.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Write images to temp files since fastembed expects paths
        // Use unique ID to avoid collisions when running in parallel
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut temp_paths: Vec<std::path::PathBuf> = Vec::with_capacity(images.len());

        for (i, image_bytes) in images.iter().enumerate() {
            // Detect image format and use appropriate extension
            let ext = crate::core::image_utils::detect_image_format(image_bytes).unwrap_or("png");
            let temp_path = temp_dir.join(format!("fastembed_img_{}_{}.{}", unique_id, i, ext));
            std::fs::write(&temp_path, image_bytes)
                .map_err(|e| EmbeddingError::ImageLoadError(e.to_string()))?;
            temp_paths.push(temp_path);
        }

        // Create embedding model
        let mut embedding = ImageEmbedding::try_new(
            ImageInitOptions::new(model).with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        let result = embedding
            .embed(temp_paths.clone(), None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()));

        // Clean up temp files
        for path in temp_paths {
            let _ = std::fs::remove_file(path);
        }

        result
    }
}

impl Default for ImageEmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sparse Text Embedding Service
// ============================================================================

/// Thread-safe sparse text embedding service
pub struct SparseEmbeddingService {
    model: Mutex<Option<(SparseModel, SparseTextEmbedding)>>,
    show_progress: bool,
}

impl SparseEmbeddingService {
    pub fn new() -> Self {
        SparseEmbeddingService {
            model: Mutex::new(None),
            show_progress: false,
        }
    }

    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Initialize with a specific model
    pub fn init(&self, model_id: &str) -> Result<(), EmbeddingError> {
        let sparse_model = ModelRegistry::find_sparse_model(model_id)
            .ok_or_else(|| EmbeddingError::ModelNotFound(model_id.to_string()))?;

        let model = SparseTextEmbedding::try_new(
            SparseInitOptions::new(sparse_model.clone())
                .with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;
        *guard = Some((sparse_model, model));

        Ok(())
    }

    /// Generate sparse embeddings for texts
    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<SparseEmbedding>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let (_, model) = guard
            .as_mut()
            .ok_or_else(|| EmbeddingError::ModelInitError("Model not initialized".to_string()))?;

        let results = model
            .embed(texts, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?;

        // Convert fastembed's sparse format to our SparseEmbedding
        Ok(results
            .into_iter()
            .map(|sparse| SparseEmbedding::new(sparse.indices, sparse.values))
            .collect())
    }

    /// Generate sparse embedding for a single text
    pub fn embed_one(&self, text: &str) -> Result<SparseEmbedding, EmbeddingError> {
        let embeddings = self.embed(vec![text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::EmbeddingFailed("No embedding generated".to_string()))
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Unload the current model
    pub fn unload(&self) -> Result<bool, EmbeddingError> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let was_loaded = guard.is_some();
        *guard = None;
        Ok(was_loaded)
    }

    /// Reload the model with a new model ID
    pub fn reload(&self, model_id: &str) -> Result<(), EmbeddingError> {
        self.unload()?;
        self.init(model_id)
    }

    /// Get the currently loaded model ID
    pub fn loaded_model_id(&self) -> Option<String> {
        self.model
            .lock()
            .ok()
            .and_then(|guard| {
                guard.as_ref().map(|(model_enum, _)| {
                    SparseTextEmbedding::list_supported_models()
                        .into_iter()
                        .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(model_enum))
                        .map(|info| info.model_code.to_string())
                        .unwrap_or_default()
                })
            })
    }

    /// Generate sparse embeddings using a specific model
    pub fn embed_with_model(
        &self,
        texts: Vec<String>,
        model: SparseModel,
    ) -> Result<Vec<SparseEmbedding>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Create embedding model
        let mut embedding = SparseTextEmbedding::try_new(
            SparseInitOptions::new(model).with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        let results = embedding
            .embed(texts, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?;

        // Convert fastembed's sparse format to our SparseEmbedding
        Ok(results
            .into_iter()
            .map(|sparse| SparseEmbedding::new(sparse.indices, sparse.values))
            .collect())
    }
}

impl Default for SparseEmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Reranking Service
// ============================================================================

/// Thread-safe text reranking service
pub struct RerankService {
    model: Mutex<Option<(RerankerModel, TextRerank)>>,
    show_progress: bool,
}

impl RerankService {
    pub fn new() -> Self {
        RerankService {
            model: Mutex::new(None),
            show_progress: false,
        }
    }

    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Initialize with a specific model
    pub fn init(&self, model_id: &str) -> Result<(), EmbeddingError> {
        let reranker_model = ModelRegistry::find_rerank_model(model_id)
            .ok_or_else(|| EmbeddingError::ModelNotFound(model_id.to_string()))?;

        let model = TextRerank::try_new(
            fastembed::RerankInitOptions::new(reranker_model.clone())
                .with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;
        *guard = Some((reranker_model, model));

        Ok(())
    }

    /// Rerank documents given a query
    pub fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        return_documents: bool,
    ) -> Result<Vec<RerankResult>, EmbeddingError> {
        if documents.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let (_, model) = guard
            .as_mut()
            .ok_or_else(|| EmbeddingError::ModelInitError("Model not initialized".to_string()))?;

        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

        let results = model
            .rerank(query, doc_refs, return_documents, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| {
                let mut result = RerankResult::new(r.index, r.score);
                if let Some(doc) = r.document {
                    result = result.with_document(doc);
                }
                result
            })
            .collect())
    }

    /// Rerank and return top K results
    pub fn rerank_top_k(
        &self,
        query: &str,
        documents: Vec<String>,
        k: usize,
        return_documents: bool,
    ) -> Result<Vec<RerankResult>, EmbeddingError> {
        let mut results = self.rerank(query, documents, return_documents)?;
        results.truncate(k);
        Ok(results)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Unload the current model
    pub fn unload(&self) -> Result<bool, EmbeddingError> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        let was_loaded = guard.is_some();
        *guard = None;
        Ok(was_loaded)
    }

    /// Reload the model with a new model ID
    pub fn reload(&self, model_id: &str) -> Result<(), EmbeddingError> {
        self.unload()?;
        self.init(model_id)
    }

    /// Get the currently loaded model ID
    pub fn loaded_model_id(&self) -> Option<String> {
        self.model
            .lock()
            .ok()
            .and_then(|guard| {
                guard.as_ref().map(|(model_enum, _)| {
                    TextRerank::list_supported_models()
                        .into_iter()
                        .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(model_enum))
                        .map(|info| info.model_code.to_string())
                        .unwrap_or_default()
                })
            })
    }

    /// Rerank documents using a specific model
    pub fn rerank_with_model(
        &self,
        query: &str,
        documents: Vec<String>,
        return_documents: bool,
        model: RerankerModel,
    ) -> Result<Vec<RerankResult>, EmbeddingError> {
        if documents.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Create rerank model
        let mut reranker = TextRerank::try_new(
            fastembed::RerankInitOptions::new(model)
                .with_show_download_progress(self.show_progress),
        )
        .map_err(|e| EmbeddingError::ModelInitError(e.to_string()))?;

        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

        let results = reranker
            .rerank(query, doc_refs, return_documents, None)
            .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| {
                let mut result = RerankResult::new(r.index, r.score);
                if let Some(doc) = r.document {
                    result = result.with_document(doc);
                }
                result
            })
            .collect())
    }
}

impl Default for RerankService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unified Multi-Model Service
// ============================================================================

/// Unified service that can handle any model type
pub enum UnifiedModel {
    Text(EmbeddingService),
    Image(ImageEmbeddingService),
    Sparse(SparseEmbeddingService),
    Rerank(RerankService),
}

/// Unified embedding service that auto-detects model type
pub struct UnifiedEmbeddingService {
    model: Mutex<Option<UnifiedModel>>,
    model_id: String,
    category: ModelCategory,
    show_progress: bool,
}

impl UnifiedEmbeddingService {
    /// Create a new unified service for the given model ID
    pub fn new(model_id: &str) -> Result<Self, EmbeddingError> {
        let category = Self::detect_category(model_id)?;

        Ok(UnifiedEmbeddingService {
            model: Mutex::new(None),
            model_id: model_id.to_string(),
            category,
            show_progress: false,
        })
    }

    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Detect the model category from the model ID
    fn detect_category(model_id: &str) -> Result<ModelCategory, EmbeddingError> {
        if ModelRegistry::find_text_model(model_id).is_some() {
            Ok(ModelCategory::TextEmbedding)
        } else if ModelRegistry::find_image_model(model_id).is_some() {
            Ok(ModelCategory::ImageEmbedding)
        } else if ModelRegistry::find_sparse_model(model_id).is_some() {
            Ok(ModelCategory::SparseTextEmbedding)
        } else if ModelRegistry::find_rerank_model(model_id).is_some() {
            Ok(ModelCategory::TextRerank)
        } else {
            Err(EmbeddingError::ModelNotFound(model_id.to_string()))
        }
    }

    /// Get the model category
    pub fn category(&self) -> ModelCategory {
        self.category
    }

    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Initialize the appropriate model
    pub fn init(&self) -> Result<(), EmbeddingError> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| EmbeddingError::LockError(e.to_string()))?;

        if guard.is_some() {
            return Ok(()); // Already initialized
        }

        let model = match self.category {
            ModelCategory::TextEmbedding => {
                let service = EmbeddingService::new().with_progress(self.show_progress);
                // Preload using the legacy type if possible
                if let Some(model_type) = ModelType::from_str(&self.model_id) {
                    service.preload(model_type)?;
                }
                UnifiedModel::Text(service)
            }
            ModelCategory::ImageEmbedding => {
                let service = ImageEmbeddingService::new().with_progress(self.show_progress);
                service.init(&self.model_id)?;
                UnifiedModel::Image(service)
            }
            ModelCategory::SparseTextEmbedding => {
                let service = SparseEmbeddingService::new().with_progress(self.show_progress);
                service.init(&self.model_id)?;
                UnifiedModel::Sparse(service)
            }
            ModelCategory::TextRerank => {
                let service = RerankService::new().with_progress(self.show_progress);
                service.init(&self.model_id)?;
                UnifiedModel::Rerank(service)
            }
        };

        *guard = Some(model);
        Ok(())
    }

    /// Check if the model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }
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

    #[test]
    fn test_image_service_creation() {
        let service = ImageEmbeddingService::new();
        assert!(!service.is_loaded());
    }

    #[test]
    fn test_sparse_service_creation() {
        let service = SparseEmbeddingService::new();
        assert!(!service.is_loaded());
    }

    #[test]
    fn test_rerank_service_creation() {
        let service = RerankService::new();
        assert!(!service.is_loaded());
    }

    #[test]
    fn test_sparse_embedding_to_dense() {
        let sparse = SparseEmbedding::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0]);
        let dense = sparse.to_dense(10);
        assert_eq!(dense.len(), 10);
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[2], 2.0);
        assert_eq!(dense[5], 3.0);
        assert_eq!(dense[1], 0.0);
    }
}

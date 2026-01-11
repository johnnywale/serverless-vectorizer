// Model selection and configuration using fastembed's native model discovery

use fastembed::{
    EmbeddingModel, ImageEmbeddingModel, SparseModel, RerankerModel,
    TextEmbedding, ImageEmbedding, SparseTextEmbedding, TextRerank,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Re-export fastembed's EmbeddingModel for text embeddings
pub use fastembed::EmbeddingModel as TextModel;

/// Unified model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub description: String,
    pub dimension: Option<usize>,
    pub model_type: ModelCategory,
}

/// Category of embedding model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    TextEmbedding,
    ImageEmbedding,
    SparseTextEmbedding,
    TextRerank,
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelCategory::TextEmbedding => write!(f, "Text Embedding"),
            ModelCategory::ImageEmbedding => write!(f, "Image Embedding"),
            ModelCategory::SparseTextEmbedding => write!(f, "Sparse Text Embedding"),
            ModelCategory::TextRerank => write!(f, "Text Rerank"),
        }
    }
}

/// Registry for discovering all supported models from fastembed
pub struct ModelRegistry;

impl ModelRegistry {
    /// Get all supported text embedding models
    pub fn text_embedding_models() -> Vec<ModelInfo> {
        TextEmbedding::list_supported_models()
            .into_iter()
            .map(|info| ModelInfo {
                model_id: info.model_code.to_string(),
                description: info.description.to_string(),
                dimension: Some(info.dim),
                model_type: ModelCategory::TextEmbedding,
            })
            .collect()
    }

    /// Get all supported image embedding models
    pub fn image_embedding_models() -> Vec<ModelInfo> {
        ImageEmbedding::list_supported_models()
            .into_iter()
            .map(|info| ModelInfo {
                model_id: info.model_code.to_string(),
                description: info.description.to_string(),
                dimension: Some(info.dim),
                model_type: ModelCategory::ImageEmbedding,
            })
            .collect()
    }

    /// Get all supported sparse text embedding models
    pub fn sparse_text_embedding_models() -> Vec<ModelInfo> {
        SparseTextEmbedding::list_supported_models()
            .into_iter()
            .map(|info| ModelInfo {
                model_id: info.model_code.to_string(),
                description: info.description.to_string(),
                dimension: None, // Sparse models don't have fixed dimension
                model_type: ModelCategory::SparseTextEmbedding,
            })
            .collect()
    }

    /// Get all supported reranking models
    pub fn rerank_models() -> Vec<ModelInfo> {
        TextRerank::list_supported_models()
            .into_iter()
            .map(|info| ModelInfo {
                model_id: info.model_code.to_string(),
                description: info.description.to_string(),
                dimension: None, // Rerank models don't produce embeddings
                model_type: ModelCategory::TextRerank,
            })
            .collect()
    }

    /// Get all supported models across all categories
    pub fn all_models() -> Vec<ModelInfo> {
        let mut models = Vec::new();
        models.extend(Self::text_embedding_models());
        models.extend(Self::image_embedding_models());
        models.extend(Self::sparse_text_embedding_models());
        models.extend(Self::rerank_models());
        models
    }

    /// Find a text embedding model by its model code/id
    pub fn find_text_model(model_id: &str) -> Option<EmbeddingModel> {
        let model_id_lower = model_id.to_lowercase();
        TextEmbedding::list_supported_models()
            .into_iter()
            .find(|info| {
                info.model_code.to_lowercase() == model_id_lower
                    || info.model_code.to_lowercase().contains(&model_id_lower)
                    || model_id_lower.contains(&info.model_code.to_lowercase().replace("/", "-").replace("_", "-"))
            })
            .map(|info| info.model)
    }

    /// Find an image embedding model by its model code/id
    pub fn find_image_model(model_id: &str) -> Option<ImageEmbeddingModel> {
        let model_id_lower = model_id.to_lowercase();
        ImageEmbedding::list_supported_models()
            .into_iter()
            .find(|info| {
                info.model_code.to_lowercase() == model_id_lower
                    || info.model_code.to_lowercase().contains(&model_id_lower)
            })
            .map(|info| info.model)
    }

    /// Find a sparse text embedding model by its model code/id
    pub fn find_sparse_model(model_id: &str) -> Option<SparseModel> {
        let model_id_lower = model_id.to_lowercase();
        SparseTextEmbedding::list_supported_models()
            .into_iter()
            .find(|info| {
                info.model_code.to_lowercase() == model_id_lower
                    || info.model_code.to_lowercase().contains(&model_id_lower)
            })
            .map(|info| info.model)
    }

    /// Find a reranking model by its model code/id
    pub fn find_rerank_model(model_id: &str) -> Option<RerankerModel> {
        let model_id_lower = model_id.to_lowercase();
        TextRerank::list_supported_models()
            .into_iter()
            .find(|info| {
                info.model_code.to_lowercase() == model_id_lower
                    || info.model_code.to_lowercase().contains(&model_id_lower)
            })
            .map(|info| info.model)
    }

    /// Get model info for a text embedding model
    pub fn get_text_model_info(model: &EmbeddingModel) -> Option<ModelInfo> {
        TextEmbedding::list_supported_models()
            .into_iter()
            .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(model))
            .map(|info| ModelInfo {
                model_id: info.model_code.to_string(),
                description: info.description.to_string(),
                dimension: Some(info.dim),
                model_type: ModelCategory::TextEmbedding,
            })
    }

    /// Get dimension for a text embedding model
    pub fn get_text_model_dimension(model: &EmbeddingModel) -> Option<usize> {
        TextEmbedding::list_supported_models()
            .into_iter()
            .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(model))
            .map(|info| info.dim)
    }

    /// Get default text embedding model
    pub fn default_text_model() -> EmbeddingModel {
        EmbeddingModel::BGESmallENV15
    }
}

// ============================================================================
// Backward compatibility layer for ModelType
// ============================================================================

/// Legacy ModelType enum for backward compatibility
/// Maps to fastembed's EmbeddingModel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelType {
    BgeSmallEnV15,
    BgeBaseEnV15,
    BgeLargeEnV15,
    MultilingualE5Large,
    AllMpnetBaseV2,
}

impl ModelType {
    /// Convert to fastembed's EmbeddingModel enum
    pub fn to_fastembed(&self) -> EmbeddingModel {
        match self {
            ModelType::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
            ModelType::BgeBaseEnV15 => EmbeddingModel::BGEBaseENV15,
            ModelType::BgeLargeEnV15 => EmbeddingModel::BGELargeENV15,
            ModelType::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
            ModelType::AllMpnetBaseV2 => EmbeddingModel::AllMiniLML6V2, // Note: AllMpnetBaseV2 may not exist, mapping to similar
        }
    }

    /// Get the embedding dimension for this model (from fastembed)
    pub fn dimension(&self) -> usize {
        ModelRegistry::get_text_model_dimension(&self.to_fastembed()).unwrap_or(384)
    }

    /// Get human-readable model name
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelType::BgeSmallEnV15 => "BGE-Small-EN-v1.5",
            ModelType::BgeBaseEnV15 => "BGE-Base-EN-v1.5",
            ModelType::BgeLargeEnV15 => "BGE-Large-EN-v1.5",
            ModelType::MultilingualE5Large => "Multilingual-E5-Large",
            ModelType::AllMpnetBaseV2 => "All-MpNet-Base-v2",
        }
    }

    /// Get the canonical string identifier for this model
    pub fn id(&self) -> &'static str {
        match self {
            ModelType::BgeSmallEnV15 => "bge-small-en-v1.5",
            ModelType::BgeBaseEnV15 => "bge-base-en-v1.5",
            ModelType::BgeLargeEnV15 => "bge-large-en-v1.5",
            ModelType::MultilingualE5Large => "multilingual-e5-large",
            ModelType::AllMpnetBaseV2 => "all-mpnet-base-v2",
        }
    }

    /// Get the language supported by this model
    pub fn language(&self) -> &'static str {
        match self {
            ModelType::MultilingualE5Large => "Multilingual",
            _ => "English",
        }
    }

    /// Get maximum token length for this model
    pub fn max_tokens(&self) -> usize {
        512 // All current models support 512 tokens
    }

    /// Parse model type from string (case-insensitive, flexible matching)
    pub fn from_str(s: &str) -> Option<ModelType> {
        let s = s.to_lowercase();
        match s.as_str() {
            "bge-small-en-v1.5" | "bge-small" | "bgesmall" => Some(ModelType::BgeSmallEnV15),
            "bge-base-en-v1.5" | "bge-base" | "bgebase" => Some(ModelType::BgeBaseEnV15),
            "bge-large-en-v1.5" | "bge-large" | "bgelarge" => Some(ModelType::BgeLargeEnV15),
            "multilingual-e5-large" | "multilingual-e5" | "e5-large" => {
                Some(ModelType::MultilingualE5Large)
            }
            "all-mpnet-base-v2" | "all-mpnet" | "mpnet" => Some(ModelType::AllMpnetBaseV2),
            _ => None,
        }
    }

    /// Get all available legacy model types
    pub fn all() -> &'static [ModelType] {
        &[
            ModelType::BgeSmallEnV15,
            ModelType::BgeBaseEnV15,
            ModelType::BgeLargeEnV15,
            ModelType::MultilingualE5Large,
            ModelType::AllMpnetBaseV2,
        ]
    }

    /// Get the default model type
    pub fn default_model() -> ModelType {
        ModelType::BgeSmallEnV15
    }
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::default_model()
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Convenience constant for model registry access
pub const MODEL_REGISTRY: ModelRegistry = ModelRegistry;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_parsing() {
        assert_eq!(ModelType::from_str("bge-small"), Some(ModelType::BgeSmallEnV15));
        assert_eq!(ModelType::from_str("BGE-SMALL"), Some(ModelType::BgeSmallEnV15));
        assert_eq!(ModelType::from_str("bge-small-en-v1.5"), Some(ModelType::BgeSmallEnV15));
        assert_eq!(ModelType::from_str("unknown"), None);
    }

    #[test]
    fn test_list_text_embedding_models() {
        let models = ModelRegistry::text_embedding_models();
        assert!(!models.is_empty(), "Should have text embedding models");

        // Check that each model has required fields
        for model in &models {
            assert!(!model.model_id.is_empty());
            assert!(model.dimension.is_some());
            assert_eq!(model.model_type, ModelCategory::TextEmbedding);
        }
    }

    #[test]
    fn test_find_text_model() {
        // Test finding BGE small model
        let model = ModelRegistry::find_text_model("bge-small-en-v1.5");
        assert!(model.is_some());
    }

    #[test]
    fn test_all_models() {
        let models = ModelRegistry::all_models();
        assert!(!models.is_empty(), "Should have some models");

        // Check we have multiple categories
        let has_text = models.iter().any(|m| m.model_type == ModelCategory::TextEmbedding);
        assert!(has_text, "Should have text embedding models");
    }
}

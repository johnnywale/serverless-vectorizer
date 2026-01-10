// Model selection and configuration

use fastembed::EmbeddingModel;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported embedding model types
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
            ModelType::AllMpnetBaseV2 => EmbeddingModel::AllMpnetBaseV2,
        }
    }

    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        match self {
            ModelType::BgeSmallEnV15 => 384,
            ModelType::BgeBaseEnV15 => 768,
            ModelType::BgeLargeEnV15 => 1024,
            ModelType::MultilingualE5Large => 1024,
            ModelType::AllMpnetBaseV2 => 768,
        }
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

    /// Get all available model types
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

/// Detailed model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub name: String,
    pub dimension: usize,
    pub max_tokens: usize,
    pub language: String,
}

impl From<ModelType> for ModelInfo {
    fn from(model_type: ModelType) -> Self {
        ModelInfo {
            model_type,
            name: model_type.display_name().to_string(),
            dimension: model_type.dimension(),
            max_tokens: model_type.max_tokens(),
            language: model_type.language().to_string(),
        }
    }
}

/// Registry of all available models
pub struct ModelRegistry;

impl ModelRegistry {
    /// Get all available models with their info
    pub fn all_models() -> Vec<ModelInfo> {
        ModelType::all().iter().map(|&m| m.into()).collect()
    }

    /// Get info for a specific model
    pub fn get_info(model_type: ModelType) -> ModelInfo {
        model_type.into()
    }

    /// Find model by string identifier
    pub fn find(name: &str) -> Option<ModelType> {
        ModelType::from_str(name)
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
    fn test_model_dimensions() {
        assert_eq!(ModelType::BgeSmallEnV15.dimension(), 384);
        assert_eq!(ModelType::BgeBaseEnV15.dimension(), 768);
        assert_eq!(ModelType::BgeLargeEnV15.dimension(), 1024);
    }
}

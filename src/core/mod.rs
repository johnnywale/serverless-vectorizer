// Core embedding functionality shared between CLI and Lambda

pub mod model;
pub mod embeddings;
pub mod similarity;
pub mod clustering;
pub mod chunking;
pub mod types;
pub mod image_utils;

#[cfg(feature = "pdf")]
pub mod pdf;

// Re-export model types
pub use model::{ModelType, ModelInfo, ModelRegistry, ModelCategory, TextModel, MODEL_REGISTRY};

// Re-export embedding services
pub use embeddings::{
    // Text embedding (existing)
    EmbeddingService, EmbeddingError, global_service, embed, embed_one,
    // Image embedding (new)
    ImageEmbeddingService,
    // Sparse embedding (new)
    SparseEmbeddingService,
    // Reranking (new)
    RerankService,
    // Unified service (new)
    UnifiedEmbeddingService, UnifiedModel,
};

// Re-export image utilities
pub use image_utils::{
    ImageError, load_image_bytes, decode_base64_image, load_image_from_file,
    load_images_bytes, is_valid_image_bytes, detect_image_format,
};

#[cfg(feature = "aws")]
pub use image_utils::s3::{
    load_image_from_s3, parse_s3_path, load_image_bytes_async, load_images_bytes_async,
};

// Re-export similarity functions
pub use similarity::{
    cosine_similarity, dot_product, euclidean_distance,
    l2_normalize, magnitude, pairwise_similarity_matrix,
    top_k_similar, compute_stats, validate_embeddings,
    SimilarityResult, EmbeddingStats, ValidationResult,
};

// Re-export clustering
pub use clustering::{
    kmeans_cluster, kmeans_cluster_with_config,
    ClusterResult, KMeansConfig,
    cluster_sizes, cluster_members, compute_inertia,
};

// Re-export chunking
pub use chunking::{
    chunk_text, chunk_text_with_config, chunk_text_detailed,
    chunk_by_sentences, estimate_tokens,
    ChunkConfig, ChunkResult,
};

// Re-export common types
pub use types::{
    EmbeddingOutput, SearchResult, SearchResponse,
    ClusterInfo, ClusterMember, ClusterResponse,
    DistanceMatrixResponse, BenchmarkResult,
    // Image types (new)
    ImageInput, ImageEmbeddingOutput,
    // Sparse types (new)
    SparseEmbedding, SparseEmbeddingOutput,
    // Rerank types (new)
    RerankResult, RerankOutput,
};

// Re-export PDF utilities (when feature enabled)
#[cfg(feature = "pdf")]
pub use pdf::{
    extract_text_from_file, extract_text_from_bytes,
    extract_text_by_pages, extract_text_by_pages_from_bytes,
    is_pdf_file, is_pdf_bytes,
    PdfDocument, PdfError,
};

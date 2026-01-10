// Core embedding functionality shared between CLI and Lambda

pub mod model;
pub mod embeddings;
pub mod similarity;
pub mod clustering;
pub mod chunking;
pub mod types;

#[cfg(feature = "pdf")]
pub mod pdf;

// Re-export model types
pub use model::{ModelType, ModelInfo, ModelRegistry, MODEL_REGISTRY};

// Re-export embedding service
pub use embeddings::{EmbeddingService, EmbeddingError, global_service, embed, embed_one};

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
};

// Re-export PDF utilities (when feature enabled)
#[cfg(feature = "pdf")]
pub use pdf::{
    extract_text_from_file, extract_text_from_bytes,
    extract_text_by_pages, extract_text_by_pages_from_bytes,
    is_pdf_file, is_pdf_bytes,
    PdfDocument, PdfError,
};

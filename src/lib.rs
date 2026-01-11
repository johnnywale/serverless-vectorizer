// Embedding Service Library
// Provides text embedding functionality for both Lambda and CLI

pub mod core;

// Lambda-specific modules (when feature enabled)
#[cfg(feature = "aws")]
mod lambda;

// Re-export core functionality for external use
pub use core::{
    // Model types
    ModelType, ModelInfo, ModelRegistry, ModelCategory, TextModel, MODEL_REGISTRY,
    // Text embedding service
    EmbeddingService, EmbeddingError, global_service, embed, embed_one,
    // Image embedding service
    ImageEmbeddingService,
    // Sparse embedding service
    SparseEmbeddingService,
    // Reranking service
    RerankService,
    // Unified service
    UnifiedEmbeddingService, UnifiedModel,
    // Image utilities
    ImageError, load_image_bytes, decode_base64_image, load_image_from_file,
    load_images_bytes, is_valid_image_bytes, detect_image_format,
    // Similarity functions
    cosine_similarity, dot_product, euclidean_distance,
    l2_normalize, magnitude, pairwise_similarity_matrix,
    top_k_similar, compute_stats, validate_embeddings,
    SimilarityResult, EmbeddingStats, ValidationResult,
    // Clustering
    kmeans_cluster, kmeans_cluster_with_config,
    ClusterResult, KMeansConfig,
    cluster_sizes, cluster_members, compute_inertia,
    // Chunking
    chunk_text, chunk_text_with_config, chunk_text_detailed,
    chunk_by_sentences, estimate_tokens,
    ChunkConfig, ChunkResult,
    // Text embedding types
    EmbeddingOutput, SearchResult, SearchResponse,
    ClusterInfo, ClusterMember, ClusterResponse,
    DistanceMatrixResponse, BenchmarkResult,
    // Image embedding types
    ImageInput, ImageEmbeddingOutput,
    // Sparse embedding types
    SparseEmbedding, SparseEmbeddingOutput,
    // Rerank types
    RerankResult, RerankOutput,
};

// Re-export PDF utilities (when feature enabled)
#[cfg(feature = "pdf")]
pub use core::{
    extract_text_from_file, extract_text_from_bytes,
    extract_text_by_pages, extract_text_by_pages_from_bytes,
    is_pdf_file, is_pdf_bytes,
    PdfDocument, PdfError,
};

// Re-export Lambda handler (when feature enabled)
#[cfg(feature = "aws")]
pub use lambda::{
    handler, Request, Response, SaveConfig, ApiGatewayResponse,
    ImageInputRequest, SparseEmbeddingResponse, RerankResponse,
};

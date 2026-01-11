// Common types and structures shared across the crate

use serde::{Deserialize, Serialize};

// ============================================================================
// Image Embedding Types
// ============================================================================

/// Image input - supports both base64 encoded data and file paths
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImageInput {
    /// Base64 encoded image data
    Base64 { base64: String },
    /// Path to image file (local or S3)
    FilePath { path: String },
    /// S3 path to image
    S3Path { s3_path: String },
}

impl ImageInput {
    /// Create from base64 string
    pub fn from_base64(data: String) -> Self {
        ImageInput::Base64 { base64: data }
    }

    /// Create from file path
    pub fn from_path(path: impl Into<String>) -> Self {
        ImageInput::FilePath { path: path.into() }
    }

    /// Create from S3 path
    pub fn from_s3(s3_path: impl Into<String>) -> Self {
        ImageInput::S3Path { s3_path: s3_path.into() }
    }
}

/// Image embedding output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingOutput {
    /// The embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Dimension of each embedding
    pub dimension: usize,
    /// Number of embeddings
    pub count: usize,
    /// Model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl ImageEmbeddingOutput {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
        let count = embeddings.len();
        ImageEmbeddingOutput {
            embeddings,
            dimension,
            count,
            model: None,
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }
}

// ============================================================================
// Sparse Embedding Types
// ============================================================================

/// Sparse embedding with indices and values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEmbedding {
    /// Token indices with non-zero values
    pub indices: Vec<usize>,
    /// Values at those indices
    pub values: Vec<f32>,
}

impl SparseEmbedding {
    pub fn new(indices: Vec<usize>, values: Vec<f32>) -> Self {
        SparseEmbedding { indices, values }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Convert to dense vector of given size
    pub fn to_dense(&self, size: usize) -> Vec<f32> {
        let mut dense = vec![0.0; size];
        for (idx, val) in self.indices.iter().zip(self.values.iter()) {
            if *idx < size {
                dense[*idx] = *val;
            }
        }
        dense
    }
}

/// Sparse embedding output with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEmbeddingOutput {
    /// Sparse embeddings
    pub embeddings: Vec<SparseEmbedding>,
    /// Number of embeddings
    pub count: usize,
    /// Model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SparseEmbeddingOutput {
    pub fn new(embeddings: Vec<SparseEmbedding>) -> Self {
        let count = embeddings.len();
        SparseEmbeddingOutput {
            embeddings,
            count,
            model: None,
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }
}

// ============================================================================
// Reranking Types
// ============================================================================

/// Single rerank result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Original index in the documents array
    pub index: usize,
    /// Relevance score (higher is more relevant)
    pub score: f32,
    /// The document text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

impl RerankResult {
    pub fn new(index: usize, score: f32) -> Self {
        RerankResult {
            index,
            score,
            document: None,
        }
    }

    pub fn with_document(mut self, document: String) -> Self {
        self.document = Some(document);
        self
    }
}

/// Rerank response with all results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankOutput {
    /// The query used for reranking
    pub query: String,
    /// Ranked results (sorted by score, descending)
    pub results: Vec<RerankResult>,
    /// Total number of documents reranked
    pub count: usize,
    /// Model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl RerankOutput {
    pub fn new(query: String, results: Vec<RerankResult>) -> Self {
        let count = results.len();
        RerankOutput {
            query,
            results,
            count,
            model: None,
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Get top K results
    pub fn top_k(&self, k: usize) -> Vec<&RerankResult> {
        self.results.iter().take(k).collect()
    }
}

/// Embedding output with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    /// The embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Dimension of each embedding
    pub dimension: usize,
    /// Number of embeddings
    pub count: usize,
    /// Model used (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl EmbeddingOutput {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);
        let count = embeddings.len();
        EmbeddingOutput {
            embeddings,
            dimension,
            count,
            model: None,
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }
}

/// Search result with text and similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Rank in results (1-indexed)
    pub rank: usize,
    /// Index in original corpus
    pub index: usize,
    /// The matched text
    pub text: String,
    /// Similarity score
    pub similarity: f32,
}

/// Search query response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// The query text
    pub query: String,
    /// Search results
    pub results: Vec<SearchResult>,
}

/// Cluster information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    /// Cluster ID
    pub id: usize,
    /// Number of members
    pub size: usize,
    /// Member details
    pub members: Vec<ClusterMember>,
}

/// Member of a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMember {
    /// Index in original data
    pub index: usize,
    /// The text
    pub text: String,
}

/// Full cluster response with all clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResponse {
    /// All clusters
    pub clusters: Vec<ClusterInfo>,
    /// Assignment for each input (which cluster)
    pub assignments: Vec<usize>,
}

impl ClusterResponse {
    /// Build from texts and cluster assignments
    pub fn from_assignments(texts: &[String], assignments: &[usize], k: usize) -> Self {
        let mut clusters: Vec<ClusterInfo> = (0..k)
            .map(|id| ClusterInfo {
                id,
                size: 0,
                members: Vec::new(),
            })
            .collect();

        for (idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < k {
                clusters[cluster_id].size += 1;
                clusters[cluster_id].members.push(ClusterMember {
                    index: idx,
                    text: texts.get(idx).cloned().unwrap_or_default(),
                });
            }
        }

        ClusterResponse {
            clusters,
            assignments: assignments.to_vec(),
        }
    }
}

/// Distance/similarity matrix response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceMatrixResponse {
    /// Original texts
    pub texts: Vec<String>,
    /// Pairwise similarity matrix
    pub similarity_matrix: Vec<Vec<f32>>,
    /// Matrix size
    pub size: usize,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Model used
    pub model: String,
    /// Number of iterations
    pub iterations: usize,
    /// Text length in words
    pub text_length: usize,
    /// Total time in milliseconds
    pub total_ms: u128,
    /// Average time per embedding in milliseconds
    pub avg_ms: f64,
    /// Throughput in embeddings per second
    pub throughput: f64,
    /// Embedding dimension
    pub dimension: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_output() {
        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let output = EmbeddingOutput::new(embeddings);

        assert_eq!(output.count, 2);
        assert_eq!(output.dimension, 3);
        assert!(output.model.is_none());
    }

    #[test]
    fn test_embedding_output_with_model() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]];
        let output = EmbeddingOutput::new(embeddings).with_model("bge-small");

        assert_eq!(output.model, Some("bge-small".to_string()));
    }

    #[test]
    fn test_cluster_response_from_assignments() {
        let texts = vec![
            "hello".to_string(),
            "world".to_string(),
            "foo".to_string(),
        ];
        let assignments = vec![0, 1, 0];
        let response = ClusterResponse::from_assignments(&texts, &assignments, 2);

        assert_eq!(response.clusters.len(), 2);
        assert_eq!(response.clusters[0].size, 2);
        assert_eq!(response.clusters[1].size, 1);
    }
}

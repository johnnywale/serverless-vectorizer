// Common types and structures shared across the crate

use serde::{Deserialize, Serialize};

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

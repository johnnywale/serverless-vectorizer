// Clustering algorithms for embeddings

use crate::core::similarity::cosine_similarity;
use serde::{Deserialize, Serialize};

/// Result of k-means clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    /// Cluster centroids
    pub centroids: Vec<Vec<f32>>,
    /// Cluster assignment for each input embedding
    pub assignments: Vec<usize>,
    /// Number of iterations run
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Configuration for k-means clustering
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub k: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence threshold (stop if no assignments change)
    pub tolerance: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        KMeansConfig {
            k: 5,
            max_iter: 100,
            tolerance: 0.0,
        }
    }
}

impl KMeansConfig {
    pub fn new(k: usize) -> Self {
        KMeansConfig {
            k,
            ..Default::default()
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

/// Perform k-means clustering on embeddings using cosine similarity
pub fn kmeans_cluster(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> ClusterResult {
    let config = KMeansConfig::new(k).with_max_iter(max_iter);
    kmeans_cluster_with_config(embeddings, &config)
}

/// Perform k-means clustering with full configuration
pub fn kmeans_cluster_with_config(
    embeddings: &[Vec<f32>],
    config: &KMeansConfig,
) -> ClusterResult {
    if embeddings.is_empty() || config.k == 0 {
        return ClusterResult {
            centroids: vec![],
            assignments: vec![],
            iterations: 0,
            converged: true,
        };
    }

    let k = config.k.min(embeddings.len());
    let dim = embeddings[0].len();

    // Initialize centroids using first k points (simple initialization)
    let mut centroids: Vec<Vec<f32>> = embeddings.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; embeddings.len()];
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Assignment step: assign each point to nearest centroid
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_sim = f32::NEG_INFINITY;

            for (c, centroid) in centroids.iter().enumerate() {
                let sim = cosine_similarity(emb, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_cluster = c;
                }
            }

            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        // Check for convergence
        if !changed {
            converged = true;
            break;
        }

        // Update step: recompute centroids as mean of assigned points
        for (c, centroid) in centroids.iter_mut().enumerate() {
            let members: Vec<&Vec<f32>> = embeddings
                .iter()
                .zip(assignments.iter())
                .filter(|(_, a)| **a == c)
                .map(|(e, _)| e)
                .collect();

            if !members.is_empty() {
                // Compute mean
                *centroid = vec![0.0; dim];
                for member in &members {
                    for (i, val) in member.iter().enumerate() {
                        centroid[i] += val;
                    }
                }
                let len = members.len() as f32;
                for val in centroid.iter_mut() {
                    *val /= len;
                }
            }
        }
    }

    ClusterResult {
        centroids,
        assignments,
        iterations,
        converged,
    }
}

/// Get cluster sizes from assignments
pub fn cluster_sizes(assignments: &[usize], k: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; k];
    for &a in assignments {
        if a < k {
            sizes[a] += 1;
        }
    }
    sizes
}

/// Get indices of members for each cluster
pub fn cluster_members(assignments: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut members = vec![Vec::new(); k];
    for (i, &a) in assignments.iter().enumerate() {
        if a < k {
            members[a].push(i);
        }
    }
    members
}

/// Compute within-cluster sum of squares (inertia)
pub fn compute_inertia(
    embeddings: &[Vec<f32>],
    centroids: &[Vec<f32>],
    assignments: &[usize],
) -> f32 {
    embeddings
        .iter()
        .zip(assignments.iter())
        .map(|(emb, &cluster)| {
            if cluster < centroids.len() {
                let centroid = &centroids[cluster];
                // Use 1 - cosine_similarity as distance
                1.0 - cosine_similarity(emb, centroid)
            } else {
                0.0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_empty() {
        let result = kmeans_cluster(&[], 3, 100);
        assert!(result.centroids.is_empty());
        assert!(result.assignments.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_kmeans_single_point() {
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let result = kmeans_cluster(&embeddings, 1, 100);
        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0], 0);
    }

    #[test]
    fn test_kmeans_k_larger_than_n() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let result = kmeans_cluster(&embeddings, 5, 100);
        // k should be clamped to n
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_cluster_sizes() {
        let assignments = vec![0, 1, 0, 2, 1, 0];
        let sizes = cluster_sizes(&assignments, 3);
        assert_eq!(sizes, vec![3, 2, 1]);
    }

    #[test]
    fn test_cluster_members() {
        let assignments = vec![0, 1, 0, 2, 1, 0];
        let members = cluster_members(&assignments, 3);
        assert_eq!(members[0], vec![0, 2, 5]);
        assert_eq!(members[1], vec![1, 4]);
        assert_eq!(members[2], vec![3]);
    }
}

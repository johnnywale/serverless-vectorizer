// Similarity and distance functions for embeddings

use serde::{Deserialize, Serialize};

/// Compute cosine similarity between two vectors
/// Returns a value between -1 and 1, where 1 means identical direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

/// Compute dot product between two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// L2 normalize a vector (convert to unit vector)
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|x| x / magnitude).collect()
    }
}

/// Compute magnitude (L2 norm) of a vector
pub fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute pairwise cosine similarity matrix for a set of embeddings
pub fn pairwise_similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in i..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim; // Symmetric
        }
    }

    matrix
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub index: usize,
    pub similarity: f32,
}

/// Find top-k most similar embeddings to a query
pub fn top_k_similar(
    query: &[f32],
    embeddings: &[Vec<f32>],
    k: usize,
) -> Vec<SimilarityResult> {
    let mut results: Vec<SimilarityResult> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| SimilarityResult {
            index: i,
            similarity: cosine_similarity(query, emb),
        })
        .collect();

    // Sort by similarity descending
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

    results.truncate(k);
    results
}

/// Statistics about a set of embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    pub count: usize,
    pub dimension: usize,
    pub min_value: f32,
    pub max_value: f32,
    pub mean_value: f32,
    pub avg_magnitude: f32,
}

/// Compute statistics for a set of embeddings
pub fn compute_stats(embeddings: &[Vec<f32>]) -> Option<EmbeddingStats> {
    if embeddings.is_empty() {
        return None;
    }

    let count = embeddings.len();
    let dimension = embeddings[0].len();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum = 0.0f64;
    let mut total_values = 0usize;

    for emb in embeddings {
        for &val in emb {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            sum += val as f64;
            total_values += 1;
        }
    }

    let mean_value = (sum / total_values as f64) as f32;

    let avg_magnitude = embeddings
        .iter()
        .map(|e| magnitude(e))
        .sum::<f32>() / count as f32;

    Some(EmbeddingStats {
        count,
        dimension,
        min_value: min_val,
        max_value: max_val,
        mean_value,
        avg_magnitude,
    })
}

/// Validate embeddings for consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub count: usize,
    pub dimension: Option<usize>,
    pub issues: Vec<String>,
}

/// Validate a set of embeddings
pub fn validate_embeddings(
    embeddings: &[Vec<f32>],
    expected_dim: Option<usize>,
) -> ValidationResult {
    let mut issues = Vec::new();
    let mut valid = true;

    if embeddings.is_empty() {
        return ValidationResult {
            valid: false,
            count: 0,
            dimension: None,
            issues: vec!["No embeddings found".to_string()],
        };
    }

    let dimension = embeddings[0].len();
    let expected = expected_dim.unwrap_or(dimension);

    for (i, emb) in embeddings.iter().enumerate() {
        // Check dimension
        if emb.len() != expected {
            issues.push(format!(
                "Embedding {} has dimension {} (expected {})",
                i,
                emb.len(),
                expected
            ));
            valid = false;
        }

        // Check for NaN or Inf
        for (j, &val) in emb.iter().enumerate() {
            if val.is_nan() {
                issues.push(format!("Embedding {}[{}] is NaN", i, j));
                valid = false;
            }
            if val.is_infinite() {
                issues.push(format!("Embedding {}[{}] is infinite", i, j));
                valid = false;
            }
        }
    }

    ValidationResult {
        valid,
        count: embeddings.len(),
        dimension: Some(dimension),
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&v1, &v2).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = l2_normalize(&v);
        let mag = magnitude(&normalized);
        assert!((mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_matrix_symmetric() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let matrix = pairwise_similarity_matrix(&embeddings);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-6);
            }
        }

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_validation() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let result = validate_embeddings(&embeddings, None);
        assert!(result.valid);
        assert_eq!(result.count, 2);
        assert_eq!(result.dimension, Some(3));
    }

    #[test]
    fn test_validation_dimension_mismatch() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0], // Wrong dimension
        ];
        let result = validate_embeddings(&embeddings, None);
        assert!(!result.valid);
        assert!(!result.issues.is_empty());
    }
}

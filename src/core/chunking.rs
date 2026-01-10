// Text chunking utilities for long document processing

use serde::{Deserialize, Serialize};

/// Configuration for text chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Maximum tokens (words) per chunk
    pub max_tokens: usize,
    /// Overlap tokens between consecutive chunks
    pub overlap: usize,
    /// Minimum chunk size (won't create chunks smaller than this)
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        ChunkConfig {
            max_tokens: 512,
            overlap: 50,
            min_chunk_size: 10,
        }
    }
}

impl ChunkConfig {
    pub fn new(max_tokens: usize, overlap: usize) -> Self {
        ChunkConfig {
            max_tokens,
            overlap,
            min_chunk_size: 10,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
        if self.overlap >= self.max_tokens {
            return Err("overlap must be less than max_tokens".to_string());
        }
        Ok(())
    }
}

/// Result of chunking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
    /// The text chunks
    pub chunks: Vec<String>,
    /// Number of chunks created
    pub count: usize,
    /// Configuration used
    pub config: ChunkConfig,
}

/// Split text into overlapping chunks based on word boundaries
pub fn chunk_text(text: &str, max_tokens: usize, overlap: usize) -> Vec<String> {
    let config = ChunkConfig::new(max_tokens, overlap);
    chunk_text_with_config(text, &config)
}

/// Split text into chunks with full configuration
pub fn chunk_text_with_config(text: &str, config: &ChunkConfig) -> Vec<String> {
    assert!(config.max_tokens > 0);
    assert!(config.overlap < config.max_tokens);
    assert!(config.min_chunk_size <= config.max_tokens);

    let words: Vec<&str> = text.split_whitespace().collect();
    let total = words.len();

    if total == 0 {
        return vec![];
    }

    if total <= config.max_tokens {
        return vec![words.join(" ")];
    }

    let step = config.max_tokens - config.overlap;
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + config.max_tokens).min(total);
        let chunk_len = end - start;
        let is_last = end == total;

        if chunk_len >= config.min_chunk_size || is_last {
            chunks.push(words[start..end].join(" "));
        }

        if is_last {
            break;
        }

        start += step;
    }

    chunks
}


/// Split text into chunks and return detailed result
pub fn chunk_text_detailed(text: &str, config: &ChunkConfig) -> ChunkResult {
    let chunks = chunk_text_with_config(text, config);
    ChunkResult {
        count: chunks.len(),
        chunks,
        config: config.clone(),
    }
}

/// Split text by sentences first, then create chunks that respect sentence boundaries
pub fn chunk_by_sentences(text: &str, max_tokens: usize, overlap: usize) -> Vec<String> {
    // Simple sentence splitting (handles ., !, ?)
    let sentences: Vec<&str> = text
        .split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_word_count = 0;

    for sentence in sentences {
        let sentence_words = sentence.split_whitespace().count();

        // If single sentence exceeds max_tokens, chunk it by words
        if sentence_words > max_tokens {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
                current_word_count = 0;
            }

            let word_chunks = chunk_text(sentence, max_tokens, overlap);
            chunks.extend(word_chunks);
            continue;
        }

        // Check if adding this sentence would exceed limit
        if current_word_count + sentence_words > max_tokens && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());

            // Start new chunk with overlap from previous
            if overlap > 0 && !chunks.is_empty() {
                let last_chunk = chunks.last().unwrap();
                let words: Vec<&str> = last_chunk.split_whitespace().collect();
                let overlap_start = words.len().saturating_sub(overlap);
                current_chunk = words[overlap_start..].join(" ");
                current_word_count = words.len() - overlap_start;
            } else {
                current_chunk = String::new();
                current_word_count = 0;
            }
        }

        // Add sentence to current chunk
        if !current_chunk.is_empty() {
            current_chunk.push_str(". ");
        }
        current_chunk.push_str(sentence);
        current_word_count += sentence_words;
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Estimate token count for text (simple word-based approximation)
pub fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_empty() {
        let chunks = chunk_text("", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_small_text() {
        let text = "Hello world";
        let chunks = chunk_text(text, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn test_chunk_with_overlap() {
        let text = "one two three four five six seven eight nine ten";

        let chunks = chunk_text_with_config(
            text,
            &ChunkConfig {
                max_tokens: 5,
                overlap: 2,
                min_chunk_size: 1,
            },
        );

        assert!(chunks.len() > 1);

        let first: Vec<&str> = chunks[0].split_whitespace().collect();
        let second: Vec<&str> = chunks[1].split_whitespace().collect();

        assert_eq!(
            &first[first.len() - 2..],
            &second[..2]
        );
    }

    #[test]
    fn test_chunk_config_validation() {
        let config = ChunkConfig::new(100, 50);
        assert!(config.validate().is_ok());

        let bad_config = ChunkConfig::new(50, 100);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("  spaces  between  "), 2);
    }
}

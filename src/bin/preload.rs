// Preload binary for all model types (text, image, sparse, rerank)
// Detects model type automatically from MODEL_ID

use fastembed::{
    ImageEmbedding, ImageInitOptions, InitOptions, RerankInitOptions, SparseInitOptions,
    SparseTextEmbedding, TextEmbedding, TextRerank,
};
use serverless_vectorizer::ModelRegistry;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_id = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("Xenova/bge-small-en-v1.5");

    println!("======================================");
    println!("Preloading model: {}", model_id);

    // Auto-detect model category and preload accordingly
    if let Some(model) = ModelRegistry::find_text_model(model_id) {
        println!("Model type: Text Embedding");
        preload_text_model(model, model_id);
    } else if let Some(model) = ModelRegistry::find_image_model(model_id) {
        println!("Model type: Image Embedding");
        preload_image_model(model, model_id);
    } else if let Some(model) = ModelRegistry::find_sparse_model(model_id) {
        println!("Model type: Sparse Text Embedding");
        preload_sparse_model(model, model_id);
    } else if let Some(model) = ModelRegistry::find_rerank_model(model_id) {
        println!("Model type: Reranking");
        preload_rerank_model(model, model_id);
    } else {
        eprintln!("Error: Unknown model '{}'", model_id);
        eprintln!("\nAvailable models by category:");
        list_all_models();
        std::process::exit(1);
    }

    println!("Done!");
}

fn preload_text_model(model: fastembed::EmbeddingModel, _model_id: &str) {
    let mut embedding = TextEmbedding::try_new(
        InitOptions::new(model).with_show_download_progress(true),
    )
    .expect("Failed to initialize text embedding model");

    // Warm up with a test embedding
    let embeddings = embedding
        .embed(vec!["test".to_string()], None)
        .expect("Failed to generate test embedding");

    println!("Embedding dimension: {}", embeddings[0].len());
}

fn preload_image_model(model: fastembed::ImageEmbeddingModel, model_id: &str) {
    let _embedding = ImageEmbedding::try_new(
        ImageInitOptions::new(model).with_show_download_progress(true),
    )
    .expect("Failed to initialize image embedding model");

    // Get dimension from model info
    let dim = ImageEmbedding::list_supported_models()
        .into_iter()
        .find(|info| info.model_code == model_id)
        .map(|info| info.dim)
        .unwrap_or(512);

    println!("Embedding dimension: {}", dim);
    println!("Note: Image model loaded (no test embedding without image file)");
}

fn preload_sparse_model(model: fastembed::SparseModel, _model_id: &str) {
    let mut embedding = SparseTextEmbedding::try_new(
        SparseInitOptions::new(model).with_show_download_progress(true),
    )
    .expect("Failed to initialize sparse embedding model");

    // Warm up with a test embedding
    let embeddings = embedding
        .embed(vec!["test".to_string()], None)
        .expect("Failed to generate test sparse embedding");

    println!(
        "Sparse embedding non-zero elements: {}",
        embeddings[0].indices.len()
    );
}

fn preload_rerank_model(model: fastembed::RerankerModel, _model_id: &str) {
    let mut reranker = TextRerank::try_new(
        RerankInitOptions::new(model).with_show_download_progress(true),
    )
    .expect("Failed to initialize reranking model");

    // Warm up with a test rerank
    let results = reranker
        .rerank("test query", vec!["test document"], true, None)
        .expect("Failed to run test rerank");

    println!("Rerank test score: {:.4}", results[0].score);
}

fn list_all_models() {
    println!("\n## Text Embedding Models:");
    for model in ModelRegistry::text_embedding_models().iter().take(10) {
        println!(
            "  - {} ({}D)",
            model.model_id,
            model.dimension.unwrap_or(0)
        );
    }
    println!("  ... and more (use list-models for full list)");

    println!("\n## Image Embedding Models:");
    for model in ModelRegistry::image_embedding_models() {
        println!(
            "  - {} ({}D)",
            model.model_id,
            model.dimension.unwrap_or(0)
        );
    }

    println!("\n## Sparse Text Embedding Models:");
    for model in ModelRegistry::sparse_text_embedding_models() {
        println!("  - {}", model.model_id);
    }

    println!("\n## Reranking Models:");
    for model in ModelRegistry::rerank_models() {
        println!("  - {}", model.model_id);
    }
}

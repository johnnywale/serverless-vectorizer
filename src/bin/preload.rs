use fastembed::{InitOptions, TextEmbedding};
use serverless_vectorizer::ModelRegistry;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_id = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("Xenova/bge-small-en-v1.5");

    // Use ModelRegistry to dynamically find the model
    let embedding_model = match ModelRegistry::find_text_model(model_id) {
        Some(model) => model,
        None => {
            eprintln!(
                "Warning: Unknown model '{}', falling back to Xenova/bge-small-en-v1.5",
                model_id
            );
            eprintln!("\nAvailable text embedding models:");
            for model_info in ModelRegistry::text_embedding_models() {
                eprintln!("  - {} ({}D)", model_info.model_id, model_info.dimension.unwrap_or(0));
            }
            ModelRegistry::default_text_model()
        }
    };

    println!("======================================");
    println!("Preloading embedding model: {}", model_id);

    let mut model =
        TextEmbedding::try_new(InitOptions::new(embedding_model).with_show_download_progress(true))
            .expect("Failed to initialize model");

    let embeddings = model
        .embed(vec!["test".to_string()], None)
        .expect("Failed to generate embedding");

    println!("Embedding dimension = {}", embeddings[0].len());
    println!("Done!");
}

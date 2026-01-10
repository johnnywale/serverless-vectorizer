use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_type = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("bge-small-en-v1.5");

    let embedding_model = match model_type {
        "bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "multilingual-e5-large" => EmbeddingModel::MultilingualE5Large,
        "all-mpnet-base-v2" => EmbeddingModel::AllMpnetBaseV2,
        _ => {
            eprintln!(
                "Warning: Unknown model '{}', falling back to bge-small-en-v1.5",
                model_type
            );
            EmbeddingModel::BGESmallENV15
        }
    };
    println!("======================================");
    println!("Using embedding model: {}", model_type);
    let mut model =
        TextEmbedding::try_new(InitOptions::new(embedding_model).with_show_download_progress(true))
            .expect("Failed to initialize model");

    let embeddings = model
        .embed(vec!["test".to_string()], None)
        .expect("Failed to generate embedding");

    println!("Embedding dimension = {}", embeddings[0].len());
    println!("Done 🎉");
}

// CLI tool to list all supported fastembed models in various formats
// Useful for CI/CD matrix generation (GitHub Actions, etc.)

use clap::{Parser, ValueEnum};
use serverless_vectorizer::ModelRegistry;

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    /// GitHub Actions matrix format (YAML)
    GithubMatrix,
    /// Simple YAML list
    Yaml,
    /// JSON array
    Json,
    /// One model per line
    Plain,
}

#[derive(Debug, Clone, ValueEnum)]
enum Category {
    /// Text embedding models
    Text,
    /// Image embedding models
    Image,
    /// Sparse text embedding models
    Sparse,
    /// Reranking models
    Rerank,
    /// All model categories
    All,
}

#[derive(Parser)]
#[command(name = "list-models")]
#[command(version = "0.1.0")]
#[command(about = "List all supported fastembed models in various formats")]
struct Cli {
    /// Output format
    #[arg(short, long, default_value = "github-matrix")]
    format: OutputFormat,

    /// Model category to list
    #[arg(short, long, default_value = "text")]
    category: Category,

    /// Platform for matrix output
    #[arg(long, default_value = "linux/amd64")]
    platform: String,

    /// Only output the matrix content (no 'strategy:' wrapper)
    #[arg(long)]
    matrix_only: bool,
}

fn model_id_to_variant(model_id: &str) -> String {
    // Convert "BAAI/bge-small-en-v1.5" -> "models--BAAI--bge-small-en-v1.5"
    format!("models--{}", model_id.replace("/", "--"))
}

fn model_id_to_type(model_id: &str) -> String {
    // Convert "BAAI/bge-small-en-v1.5" -> "bge-small-en-v1.5"
    model_id
        .split('/')
        .last()
        .unwrap_or(model_id)
        .to_string()
}

fn main() {
    let cli = Cli::parse();

    // Collect models based on category
    let models = match cli.category {
        Category::Text => ModelRegistry::text_embedding_models(),
        Category::Image => ModelRegistry::image_embedding_models(),
        Category::Sparse => ModelRegistry::sparse_text_embedding_models(),
        Category::Rerank => ModelRegistry::rerank_models(),
        Category::All => ModelRegistry::all_models(),
    };

    match cli.format {
        OutputFormat::GithubMatrix => {
            if !cli.matrix_only {
                println!("strategy:");
                println!("  matrix:");
                println!("    include:");
            } else {
                println!("matrix:");
                println!("  include:");
            }
            for model in &models {
                let variant = model_id_to_variant(&model.model_id);
                let model_type = model_id_to_type(&model.model_id);
                let indent = if cli.matrix_only { "    " } else { "      " };
                println!("{}- variant: {}", indent, variant);
                println!("{}  model_type: {}", indent, model_type);
                println!("{}  model_id: {}", indent, model.model_id);
                if let Some(dim) = model.dimension {
                    println!("{}  dimension: {}", indent, dim);
                }
                println!("{}  platform: {}", indent, cli.platform);
            }
        }

        OutputFormat::Yaml => {
            println!("models:");
            for model in &models {
                println!("  - id: \"{}\"", model.model_id);
                println!("    type: \"{}\"", model_id_to_type(&model.model_id));
                if let Some(dim) = model.dimension {
                    println!("    dimension: {}", dim);
                }
                println!("    category: {:?}", model.model_type);
                println!("    description: \"{}\"", model.description.replace("\"", "\\\""));
            }
        }

        OutputFormat::Json => {
            let json_models: Vec<serde_json::Value> = models
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "id": m.model_id,
                        "type": model_id_to_type(&m.model_id),
                        "variant": model_id_to_variant(&m.model_id),
                        "dimension": m.dimension,
                        "category": format!("{:?}", m.model_type),
                        "description": m.description
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_models).unwrap());
        }

        OutputFormat::Plain => {
            for model in &models {
                if let Some(dim) = model.dimension {
                    println!("{}\t{}D\t{:?}", model.model_id, dim, model.model_type);
                } else {
                    println!("{}\t-\t{:?}", model.model_id, model.model_type);
                }
            }
        }
    }

    // Print summary to stderr
    eprintln!("\n# Total: {} models", models.len());
}

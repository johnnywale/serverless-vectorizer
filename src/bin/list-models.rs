// CLI tool to list all supported fastembed models in various formats
// Useful for CI/CD matrix generation (GitHub Actions, etc.)

use clap::{Parser, ValueEnum};
use serverless_vectorizer::{ModelInfo, ModelRegistry};

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    /// Both Markdown and GitHub Actions matrix format (default)
    All,
    /// Markdown table format
    Markdown,
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
    #[arg(short, long, default_value = "all")]
    format: OutputFormat,

    /// Model category to list
    #[arg(short, long, default_value = "all")]
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
    model_id.split('/').last().unwrap_or(model_id).to_string()
}

fn model_id_to_display_name(model_id: &str) -> String {
    // Convert "BAAI/bge-small-en-v1.5" -> "BGE-Small-EN-v1.5"
    let short_id = model_id.split('/').last().unwrap_or(model_id);

    // Split by hyphens and capitalize appropriately
    short_id
        .split('-')
        .map(|part| {
            // Handle version numbers (v1.5, v2, etc.)
            if part.starts_with('v')
                && part.len() > 1
                && part
                    .chars()
                    .skip(1)
                    .next()
                    .map_or(false, |c| c.is_numeric())
            {
                part.to_string()
            }
            // Handle common abbreviations that should be uppercase
            else if part.eq_ignore_ascii_case("en")
                || part.eq_ignore_ascii_case("e5")
                || part.eq_ignore_ascii_case("bge")
                || part.eq_ignore_ascii_case("gte")
                || part.eq_ignore_ascii_case("jina")
                || part.eq_ignore_ascii_case("xl")
                || part.eq_ignore_ascii_case("zh")
                || part.eq_ignore_ascii_case("ml")
                || part.eq_ignore_ascii_case("l6")
                || part.eq_ignore_ascii_case("l12")
                || part.eq_ignore_ascii_case("mpnet")
                || part.eq_ignore_ascii_case("minilm")
            {
                part.to_uppercase()
            }
            // Capitalize first letter of other parts
            else {
                let mut chars = part.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join("-")
}



fn print_markdown_table(models: &[ModelInfo]) {
    use std::collections::HashMap;

    // Group models by category
    let mut grouped: HashMap<String, Vec<&ModelInfo>> = HashMap::new();
    for model in models {
        let category = format!("{}", model.model_type);
        grouped.entry(category).or_default().push(model);
    }

    // Define category order
    let category_order = [
        ("Text Embedding", "Text Embedding Models"),
        ("Image Embedding", "Image Embedding Models"),
        ("Sparse Text Embedding", "Sparse Text Embedding Models"),
        ("Text Rerank", "Reranking Models"),
    ];

    println!("## Supported Models\n");

    for (category_key, category_title) in &category_order {
        if let Some(category_models) = grouped.get(*category_key) {
            println!("### {}\n", category_title);
            println!("| Model | Model ID | Dimension | Description |");
            println!("|-------|----------|-----------|-------------|");

            for model in category_models {
                let display_name = model_id_to_display_name(&model.model_id);
                let short_id = model.model_id.as_str();
                let dimension = model
                    .dimension
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "-".to_string());

                println!(
                    "| {} | `{}` | {} | {} |",
                    display_name, short_id, dimension, model.description
                );
            }
            println!();
        }
    }
}

fn print_github_matrix(models: &[ModelInfo], platform: &str, matrix_only: bool) {
    if !matrix_only {
        println!("strategy:");
        println!("  matrix:");
        println!("    include:");
    } else {
        println!("matrix:");
        println!("  include:");
    }
    for model in models {
        let variant = model_id_to_variant(&model.model_id);
        let model_type = model_id_to_type(&model.model_id);
        let indent = if matrix_only { "    " } else { "      " };
        println!("{}- variant: {}", indent, variant);
        println!("{}  model_type: {}", indent, model_type);
        println!("{}  model_id: {}", indent, model.model_id);
        if let Some(dim) = model.dimension {
            println!("{}  dimension: {}", indent, dim);
        }
        println!("{}  platform: {}", indent, platform);
    }
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
        OutputFormat::All => {
            print_markdown_table(&models);
            println!("\n---\n");
            print_github_matrix(&models, &cli.platform, cli.matrix_only);
        }

        OutputFormat::Markdown => {
            print_markdown_table(&models);
        }

        OutputFormat::GithubMatrix => {
            print_github_matrix(&models, &cli.platform, cli.matrix_only);
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
                println!(
                    "    description: \"{}\"",
                    model.description.replace("\"", "\\\"")
                );
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

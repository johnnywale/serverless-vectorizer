// CLI tool for text embedding operations
// Uses shared core functionality from the embedding_service library

use clap::{Parser, Subcommand, ValueEnum};
use serverless_vectorizer::{
    // Core functionality
    ModelType, EmbeddingService,
    cosine_similarity, l2_normalize, pairwise_similarity_matrix,
    top_k_similar, compute_stats, validate_embeddings,
    kmeans_cluster, chunk_text,
    // Types
    EmbeddingOutput, SearchResult, SearchResponse,
    ClusterResponse, DistanceMatrixResponse, BenchmarkResult,
};
#[cfg(feature = "pdf")]
use embedding_service::{extract_text_from_file, is_pdf_file};
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Instant;

/// CLI model choice enum (maps to core ModelType)
#[derive(Debug, Clone, ValueEnum)]
enum ModelChoice {
    BgeSmall,
    BgeBase,
    BgeLarge,
    MultilingualE5,
    AllMpnet,
}

impl ModelChoice {
    fn to_model_type(&self) -> ModelType {
        match self {
            ModelChoice::BgeSmall => ModelType::BgeSmallEnV15,
            ModelChoice::BgeBase => ModelType::BgeBaseEnV15,
            ModelChoice::BgeLarge => ModelType::BgeLargeEnV15,
            ModelChoice::MultilingualE5 => ModelType::MultilingualE5Large,
            ModelChoice::AllMpnet => ModelType::AllMpnetBaseV2,
        }
    }
}

#[derive(Parser)]
#[command(name = "embed-cli")]
#[command(author = "RAG Lambda")]
#[command(version = "0.1.0")]
#[command(about = "Text embedding CLI tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate embeddings for text
    Embed {
        #[arg(short, long)]
        text: Option<String>,
        #[arg(short, long)]
        file: Option<PathBuf>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "json")]
        format: String,
        #[arg(long)]
        pretty: bool,
        #[arg(long)]
        vectors_only: bool,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// Compute similarity between two texts
    Similarity {
        #[arg(long)]
        text1: String,
        #[arg(long)]
        text2: String,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// Embed multiple texts from a file
    Batch {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "lines")]
        input_format: String,
        #[arg(long)]
        pretty: bool,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// Show model information
    Info,

    /// Search for similar texts in a corpus
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long)]
        corpus: PathBuf,
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
        #[arg(long, default_value = "texts")]
        corpus_format: String,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// Compute pairwise similarity matrix
    DistanceMatrix {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "lines")]
        input_format: String,
        #[arg(long)]
        pretty: bool,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// L2 normalize embeddings
    Normalize {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        pretty: bool,
    },

    /// Show embedding statistics
    Stats {
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Validate embedding file
    Validate {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(long)]
        expected_dim: Option<usize>,
    },

    /// Split text into chunks
    Chunk {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        #[arg(long, default_value = "50")]
        overlap: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        pretty: bool,
    },

    /// Benchmark embedding performance
    Benchmark {
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        #[arg(long, default_value = "50")]
        text_length: usize,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// Cluster texts using k-means
    Cluster {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        k: usize,
        #[arg(long, default_value = "100")]
        max_iter: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "lines")]
        input_format: String,
        #[arg(long)]
        pretty: bool,
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },

    /// Extract text from PDF and optionally embed (requires 'pdf' feature)
    #[cfg(feature = "pdf")]
    Pdf {
        /// Input PDF file
        #[arg(short, long)]
        file: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Generate embeddings for extracted text
        #[arg(long)]
        embed: bool,

        /// Chunk the PDF text before embedding
        #[arg(long)]
        chunk: bool,

        /// Maximum tokens per chunk (when --chunk is used)
        #[arg(long, default_value = "512")]
        max_tokens: usize,

        /// Overlap tokens between chunks (when --chunk is used)
        #[arg(long, default_value = "50")]
        overlap: usize,

        /// Pretty print JSON output
        #[arg(long)]
        pretty: bool,

        /// Embedding model to use (when --embed is used)
        #[arg(long, value_enum, default_value = "bge-small")]
        model: ModelChoice,
    },
}

// Global embedding service
static SERVICE: std::sync::LazyLock<EmbeddingService> =
    std::sync::LazyLock::new(|| EmbeddingService::new().with_progress(true));

fn get_input(text: Option<String>, file: Option<PathBuf>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if let Some(t) = text {
        Ok(vec![t])
    } else if let Some(path) = file {
        // Check if file is a PDF (only when pdf feature enabled)
        #[cfg(feature = "pdf")]
        if is_pdf_file(&path) {
            eprintln!("Detected PDF file, extracting text...");
            let pdf_doc = extract_text_from_file(&path)?;
            eprintln!("Extracted {} characters from PDF", pdf_doc.text.len());
            return Ok(vec![pdf_doc.text]);
        }

        let content = fs::read_to_string(&path)?;
        if let Ok(texts) = serde_json::from_str::<Vec<String>>(&content) {
            Ok(texts)
        } else {
            Ok(vec![content])
        }
    } else {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Ok(vec![buffer])
    }
}

fn parse_texts_from_file(path: &PathBuf, format: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let texts: Vec<String> = match format {
        "json" => serde_json::from_str(&content)?,
        _ => content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect(),
    };
    Ok(texts)
}

fn write_output(content: &str, output: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = output {
        fs::write(path, content)?;
    } else {
        println!("{}", content);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Embed { text, file, output, format, pretty, vectors_only, model } => {
            let model_type = model.to_model_type();
            let texts = get_input(text, file)?;
            let embeddings = SERVICE.embed(texts, model_type)?;

            if format == "binary" {
                let mut bytes = Vec::new();
                for emb in &embeddings {
                    for &val in emb {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                }
                if let Some(path) = output {
                    fs::write(path, &bytes)?;
                } else {
                    eprintln!("Binary format requires --output file");
                    std::process::exit(1);
                }
            } else {
                let result = if vectors_only {
                    serde_json::to_value(&embeddings)?
                } else {
                    let out = EmbeddingOutput::new(embeddings).with_model(model_type.display_name());
                    serde_json::to_value(&out)?
                };
                let output_str = if pretty {
                    serde_json::to_string_pretty(&result)?
                } else {
                    serde_json::to_string(&result)?
                };
                write_output(&output_str, output)?;
            }
        }

        Commands::Similarity { text1, text2, model } => {
            let model_type = model.to_model_type();
            let embeddings = SERVICE.embed(vec![text1, text2], model_type)?;
            let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
            println!("{:.6}", similarity);
        }

        Commands::Batch { file, output, input_format, pretty, model } => {
            let model_type = model.to_model_type();
            let texts = parse_texts_from_file(&file, &input_format)?;

            if texts.is_empty() {
                eprintln!("No texts found in input file");
                std::process::exit(1);
            }

            eprintln!("Embedding {} texts...", texts.len());
            let embeddings = SERVICE.embed(texts, model_type)?;
            eprintln!("Done.");

            let out = EmbeddingOutput::new(embeddings).with_model(model_type.display_name());
            let output_str = if pretty {
                serde_json::to_string_pretty(&out)?
            } else {
                serde_json::to_string(&out)?
            };
            write_output(&output_str, output)?;
        }

        Commands::Info => {
            println!("Embedding Model Information");
            println!("===========================\n");
            println!("Available Models:");
            for model_type in ModelType::all() {
                println!("  {:20} {:25} ({} dim, {})",
                    model_type.id(),
                    model_type.display_name(),
                    model_type.dimension(),
                    model_type.language()
                );
            }
            println!("\nDefault: {}", ModelType::default().id());
            println!("Max Tokens: 512");
            println!("Provider: BAAI/Sentence-Transformers (via fastembed)");
            println!("\nModels are downloaded on first use if not cached.");

            #[cfg(feature = "pdf")]
            println!("\nPDF support: enabled");
            #[cfg(not(feature = "pdf"))]
            println!("\nPDF support: disabled (compile with --features pdf)");
        }

        Commands::Search { query, corpus, top_k, corpus_format, model } => {
            let model_type = model.to_model_type();
            let corpus_content = fs::read_to_string(&corpus)?;

            let (corpus_texts, corpus_embeddings): (Vec<String>, Vec<Vec<f32>>) =
                if corpus_format == "embeddings" {
                    let data: serde_json::Value = serde_json::from_str(&corpus_content)?;
                    let texts: Vec<String> = data.get("texts")
                        .and_then(|t| serde_json::from_value(t.clone()).ok())
                        .unwrap_or_default();
                    let embeddings: Vec<Vec<f32>> = data.get("embeddings")
                        .and_then(|e| serde_json::from_value(e.clone()).ok())
                        .unwrap_or_default();
                    (texts, embeddings)
                } else {
                    let texts: Vec<String> = if corpus_content.trim().starts_with('[') {
                        serde_json::from_str(&corpus_content)?
                    } else {
                        corpus_content.lines()
                            .filter(|l| !l.trim().is_empty())
                            .map(|l| l.to_string())
                            .collect()
                    };
                    eprintln!("Embedding {} corpus texts...", texts.len());
                    let embeddings = SERVICE.embed(texts.clone(), model_type)?;
                    (texts, embeddings)
                };

            if corpus_texts.is_empty() {
                eprintln!("No texts found in corpus");
                std::process::exit(1);
            }

            let query_emb = SERVICE.embed_one(&query, model_type)?;
            let similar = top_k_similar(&query_emb, &corpus_embeddings, top_k);

            let results: Vec<SearchResult> = similar.iter().enumerate()
                .map(|(rank, r)| SearchResult {
                    rank: rank + 1,
                    index: r.index,
                    text: corpus_texts.get(r.index).cloned().unwrap_or_default(),
                    similarity: r.similarity,
                })
                .collect();

            let response = SearchResponse { query, results };
            println!("{}", serde_json::to_string_pretty(&response)?);
        }

        Commands::DistanceMatrix { file, output, input_format, pretty, model } => {
            let model_type = model.to_model_type();
            let texts = parse_texts_from_file(&file, &input_format)?;

            if texts.is_empty() {
                eprintln!("No texts found in input file");
                std::process::exit(1);
            }

            eprintln!("Embedding {} texts...", texts.len());
            let embeddings = SERVICE.embed(texts.clone(), model_type)?;
            let matrix = pairwise_similarity_matrix(&embeddings);

            let response = DistanceMatrixResponse {
                texts,
                similarity_matrix: matrix,
                size: embeddings.len(),
            };

            let output_str = if pretty {
                serde_json::to_string_pretty(&response)?
            } else {
                serde_json::to_string(&response)?
            };
            write_output(&output_str, output)?;
        }

        Commands::Normalize { input, output, pretty } => {
            let content = fs::read_to_string(&input)?;
            let data: serde_json::Value = serde_json::from_str(&content)?;

            let embeddings: Vec<Vec<f32>> = if let Some(emb) = data.get("embeddings") {
                serde_json::from_value(emb.clone())?
            } else {
                serde_json::from_value(data)?
            };

            let normalized: Vec<Vec<f32>> = embeddings.iter()
                .map(|e| l2_normalize(e))
                .collect();

            let out = EmbeddingOutput::new(normalized);
            let result = serde_json::json!({
                "embeddings": out.embeddings,
                "dimension": out.dimension,
                "count": out.count,
                "normalized": true
            });

            let output_str = if pretty {
                serde_json::to_string_pretty(&result)?
            } else {
                serde_json::to_string(&result)?
            };
            write_output(&output_str, output)?;
        }

        Commands::Stats { input } => {
            let content = fs::read_to_string(&input)?;
            let data: serde_json::Value = serde_json::from_str(&content)?;

            let embeddings: Vec<Vec<f32>> = if let Some(emb) = data.get("embeddings") {
                serde_json::from_value(emb.clone())?
            } else {
                serde_json::from_value(data)?
            };

            if let Some(stats) = compute_stats(&embeddings) {
                println!("Embedding Statistics");
                println!("====================");
                println!("Count: {}", stats.count);
                println!("Dimension: {}", stats.dimension);
                println!("\nGlobal Statistics:");
                println!("  Min value: {:.6}", stats.min_value);
                println!("  Max value: {:.6}", stats.max_value);
                println!("  Mean value: {:.6}", stats.mean_value);
                println!("  Avg magnitude (L2 norm): {:.6}", stats.avg_magnitude);
            } else {
                eprintln!("No embeddings found in input file");
                std::process::exit(1);
            }
        }

        Commands::Validate { input, expected_dim } => {
            let content = fs::read_to_string(&input)?;
            let data: serde_json::Value = serde_json::from_str(&content)?;

            let embeddings: Vec<Vec<f32>> = if let Some(emb) = data.get("embeddings") {
                serde_json::from_value(emb.clone())?
            } else {
                serde_json::from_value(data)?
            };

            let result = validate_embeddings(&embeddings, expected_dim);

            if result.valid {
                println!("Valid: true");
                println!("Count: {}", result.count);
                if let Some(dim) = result.dimension {
                    println!("Dimension: {}", dim);
                }
            } else {
                println!("Valid: false");
                println!("Issues:");
                for issue in &result.issues {
                    println!("  - {}", issue);
                }
                std::process::exit(1);
            }
        }

        Commands::Chunk { file, max_tokens, overlap, output, pretty } => {
            let content = fs::read_to_string(&file)?;
            let chunks = chunk_text(&content, max_tokens, overlap);

            let result = serde_json::json!({
                "chunks": chunks,
                "count": chunks.len(),
                "max_tokens": max_tokens,
                "overlap": overlap
            });

            let output_str = if pretty {
                serde_json::to_string_pretty(&result)?
            } else {
                serde_json::to_string(&result)?
            };
            write_output(&output_str, output)?;
        }

        Commands::Benchmark { iterations, text_length, model } => {
            let model_type = model.to_model_type();

            let sample_words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                "hello", "world", "machine", "learning", "embedding", "vector",
                "semantic", "search", "natural", "language", "processing", "model"];
            let sample_text: String = (0..text_length)
                .map(|i| sample_words[i % sample_words.len()])
                .collect::<Vec<_>>()
                .join(" ");

            println!("Benchmark Configuration");
            println!("=======================");
            println!("Model: {}", model_type.display_name());
            println!("Iterations: {}", iterations);
            println!("Text length: {} words\n", text_length);

            eprintln!("Warming up model...");
            let _ = SERVICE.embed(vec![sample_text.clone()], model_type)?;

            eprintln!("Running benchmark...");
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = SERVICE.embed(vec![sample_text.clone()], model_type)?;
            }
            let elapsed = start.elapsed();

            let result = BenchmarkResult {
                model: model_type.display_name().to_string(),
                iterations,
                text_length,
                total_ms: elapsed.as_millis(),
                avg_ms: elapsed.as_millis() as f64 / iterations as f64,
                throughput: 1000.0 / (elapsed.as_millis() as f64 / iterations as f64),
                dimension: model_type.dimension(),
            };

            println!("Results");
            println!("=======");
            println!("Total time: {} ms", result.total_ms);
            println!("Avg per embedding: {:.2} ms", result.avg_ms);
            println!("Throughput: {:.2} embeddings/sec", result.throughput);
            println!("Dimension: {}", result.dimension);
        }

        Commands::Cluster { file, k, max_iter, output, input_format, pretty, model } => {
            let model_type = model.to_model_type();
            let texts = parse_texts_from_file(&file, &input_format)?;

            if texts.is_empty() {
                eprintln!("No texts found in input file");
                std::process::exit(1);
            }

            if k == 0 || k > texts.len() {
                eprintln!("k must be between 1 and {}", texts.len());
                std::process::exit(1);
            }

            eprintln!("Embedding {} texts...", texts.len());
            let embeddings = SERVICE.embed(texts.clone(), model_type)?;

            eprintln!("Clustering into {} clusters...", k);
            let cluster_result = kmeans_cluster(&embeddings, k, max_iter);

            let response = ClusterResponse::from_assignments(&texts, &cluster_result.assignments, k);

            let output_str = if pretty {
                serde_json::to_string_pretty(&response)?
            } else {
                serde_json::to_string(&response)?
            };
            write_output(&output_str, output)?;
        }

        #[cfg(feature = "pdf")]
        Commands::Pdf { file, output, embed, chunk, max_tokens, overlap, pretty, model } => {
            // Extract text from PDF
            eprintln!("Extracting text from PDF: {:?}", file);
            let pdf_doc = extract_text_from_file(&file)?;

            eprintln!("Extracted {} characters from {} pages",
                pdf_doc.text.len(), pdf_doc.page_count);

            if embed {
                let model_type = model.to_model_type();

                // Determine texts to embed
                let texts: Vec<String> = if chunk {
                    eprintln!("Chunking text (max_tokens={}, overlap={})...", max_tokens, overlap);
                    chunk_text(&pdf_doc.text, max_tokens, overlap)
                } else {
                    vec![pdf_doc.text.clone()]
                };

                if texts.is_empty() {
                    eprintln!("No text extracted from PDF");
                    std::process::exit(1);
                }

                eprintln!("Embedding {} text segment(s)...", texts.len());
                let embeddings = SERVICE.embed(texts.clone(), model_type)?;

                let result = serde_json::json!({
                    "source": file.to_string_lossy(),
                    "page_count": pdf_doc.page_count,
                    "texts": texts,
                    "embeddings": embeddings,
                    "dimension": embeddings.first().map(|e| e.len()).unwrap_or(0),
                    "count": embeddings.len(),
                    "model": model_type.display_name(),
                    "chunked": chunk
                });

                let output_str = if pretty {
                    serde_json::to_string_pretty(&result)?
                } else {
                    serde_json::to_string(&result)?
                };
                write_output(&output_str, output)?;
            } else {
                // Just output extracted text
                let result = serde_json::json!({
                    "source": file.to_string_lossy(),
                    "page_count": pdf_doc.page_count,
                    "word_count": pdf_doc.word_count(),
                    "text": pdf_doc.text
                });

                let output_str = if pretty {
                    serde_json::to_string_pretty(&result)?
                } else {
                    serde_json::to_string(&result)?
                };
                write_output(&output_str, output)?;
            }
        }
    }

    Ok(())
}

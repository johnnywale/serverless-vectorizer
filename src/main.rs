// src/main.rs
use lambda_runtime::{Error, service_fn};
use serverless_vectorizer::handler;

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}

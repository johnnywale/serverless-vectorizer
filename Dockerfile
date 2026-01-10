FROM   johnnywalee/aws-lambda-rust:al2023   AS chef


FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

WORKDIR /app

# Build for x86_64-unknown-linux-gnu (glibc) - NOT musl
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN cargo build --release --bin bootstrap
RUN cargo build --release --bin preload
# The binary will be at target/release/bootstrap (or your binary name)
RUN cp target/release/bootstrap bootstrap 2>/dev/null || \
    cp target/release/embedding_service bootstrap

RUN cp target/release/preload preload

# Strip the binary
RUN strip bootstrap

# Use AL2 runtime (glibc-based) instead of AL2023
FROM public.ecr.aws/lambda/provided:al2023

# Copy the binary
COPY --from=builder /app/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

COPY --from=builder /app/preload ${LAMBDA_RUNTIME_DIR}/preload



RUN chmod +x ${LAMBDA_RUNTIME_DIR}/bootstrap


CMD [ "bootstrap" ]

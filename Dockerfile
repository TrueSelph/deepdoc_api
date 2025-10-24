# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build system dependencies
RUN apt-get update && apt-get install -y \
	gcc \
	g++ \
	make \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies globally (not with --user)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download RapidOCR models during build to avoid permission issues
RUN mkdir -p /tmp/rapidocr_cache && \
    RAPIDOCR_CACHE_DIR=/tmp/rapidocr_cache python -c "from rapidocr import RapidOCR; ocr = RapidOCR(); print('RapidOCR models downloaded successfully')"

# Runtime stage
FROM python:3.10-slim AS rapidocr-runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
	poppler-utils \
	tesseract-ocr \
	tesseract-ocr-eng \
	tesseract-ocr-deu \
	tesseract-ocr-fra \
	tesseract-ocr-spa \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local /usr/local

# Copy RapidOCR models from builder stage
# Copy RapidOCR models to a user-writable directory
COPY --from=builder /usr/local/lib/python3.10/site-packages/rapidocr/models/ /app/models_cache/models
COPY --from=builder /tmp/rapidocr_cache/ /app/models_cache/rapidocr_cache

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p ./uploads ./processed ./app/temp ./app/models_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THUMBNAILS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
# Configure RapidOCR to download models to a user-writable directory
ENV RAPIDOCR_CACHE_DIR=/app/models_cache

# Expose port
EXPOSE 8991

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
	CMD curl -f http://localhost:8991/health || exit 1

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8991", "--workers", "2"]


# # curl -X POST "http://localhost:8991/upload_and_chunk" -F "files=@QualiTEST Prospective Students Information January 2026.pdf"

curl "http://localhost:8991/job/f380c94b-5110-4d92-9345-012efe2cfa0b"

# curl -X POST "http://localhost:8991/upload_and_chunk" \
#   -F "files=@QualiTEST Prospective Students Information January 2026.pdf" \
#   -F "chunker_type=toc"

#   #   -F "toc_params={\"section_pattern\": \"^(\\d+(?:\\.\\d+)*)\", \"approved_sections\": [\"1\", \"2\", \"3\"]}"

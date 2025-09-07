# Build stage
FROM python:3.10-slim as builder
WORKDIR /app

# Install build system dependencies
RUN apt-get update && apt-get install -y \
	gcc \
	g++ \
	make \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies globally (not with --user)
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app

# Install runtime system dependencies INCLUDING Python UNO bindings
RUN apt-get update && apt-get install -y \
	poppler-utils \
	tesseract-ocr \
	tesseract-ocr-eng \
	tesseract-ocr-deu \
	tesseract-ocr-fra \
	tesseract-ocr-spa \
	libgl1 \
	libreoffice \
	libreoffice-script-provider-python \
	python3-uno \
	libglib2.0-0 \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local /usr/local

# Set up LibreOffice headless mode and UNO environment
ENV PYTHONPATH=/app:/usr/lib/python3/dist-packages
ENV UNO_PATH=/usr/lib/libreoffice/program

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p ./uploads ./app/temp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THUMBNAILS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Expose ports
EXPOSE 8991 5555

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
	CMD curl -f http://localhost:8991/health || exit 1

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

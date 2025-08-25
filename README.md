# DeepDoc API Service - powered by Docling

A high-performance document processing API service built with FastAPI that uses Docling for intelligent document parsing, chunking, and embedding generation.

## Features

- **Multi-format Support**: Process PDF, DOCX, PPTX, HTML, TXT, JPG, JPEG, and PNG files
- **Intelligent Chunking**: Uses Docling's HybridChunker for semantic text chunking
- **Embedding Generation**: Integrated support for external embedding services
- **URL Processing**: Download and process files from URLs and Google Drive links
- **Background Processing**: Asynchronous job processing with status tracking
- **Callback Support**: Webhook notifications for job completion
- **Redis Integration**: Persistent job status and result storage

## API Endpoints

### POST `/upload_and_chunk`
Process documents from file uploads or URLs.

**Parameters:**
- `files`: List of uploaded files (optional)
- `urls`: List of URLs to process (optional)
- `from_page`: Starting page number (default: 0)
- `to_page`: Ending page number (default: 100000)
- `lang`: Language for processing (default: "english")
- `with_embeddings`: Generate embeddings for chunks (default: false)
- `callback_url`: Webhook URL for job completion notifications

**Response:**
```json
{"job_id": "uuid-string"}
```

### GET `/job/{job_id}`
Check the status of a processing job.

**Response:**
```json
{
  "status": "pending|processing|completed|failed|cancelled",
  "result": [/* array of chunks */],
  "error": "error message if failed"
}
```

### POST `/job/{job_id}/cancel`
Cancel a running job.

### GET `/health`
Comprehensive health check endpoint.

### GET `/liveness`
Lightweight health check for Kubernetes liveness probes.

## Installation

### Docker (Recommended)

```bash
# Build the image
docker build -t docling-api .

# Run the container
docker run -p 8991:8991 \
  -e EMBEDDING_SERVICE_URL="your_embedding_url" \
  -e EMBEDDING_SERVICE_API_KEY="your_api_key" \
  -v ./uploads:/app/uploads \
  docling-api
```

### Docker Compose

```bash
# Start with docker-compose
docker-compose up -d
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export EMBEDDING_SERVICE_URL="your_embedding_url"
export EMBEDDING_SERVICE_API_KEY="your_api_key"
export REDIS_URL="redis://localhost:6379/0"

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8991
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Host to bind the server to | `0.0.0.0` |
| `API_PORT` | Port to run the server on | `8991` |
| `EMBEDDING_SERVICE_URL` | URL of the embedding service | - |
| `EMBEDDING_SERVICE_API_KEY` | API key for embedding service | - |
| `EMBEDDING_MODEL` | Model to use for embeddings | `intfloat/multilingual-e5-large-instruct` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `UPLOAD_DIR` | Directory for uploaded files | `./uploads` |
| `PROCESSED_DIR` | Directory for processed files | `./processed` |
| `MAX_FILE_SIZE` | Maximum file size in bytes | `52428800` (50MB) |

## Usage Examples

### Process Uploaded Files

```bash
curl -X POST "http://localhost:8991/upload_and_chunk" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "from_page=0" \
  -F "to_page=100000" \
  -F "lang=english" \
  -F "with_embeddings=true"
```

### Process URLs

```bash
curl -X POST "http://localhost:8991/upload_and_chunk" \
  -F "urls=https://example.com/document.pdf" \
  -F "urls=https://drive.google.com/file/d/GOOGLE_DRIVE_ID/view" \
  -F "with_embeddings=true"
```

### Check Job Status

```bash
curl "http://localhost:8991/job/12345678-1234-1234-1234-123456789012"
```

### Cancel a Job

```bash
curl -X POST "http://localhost:8991/job/12345678-1234-1234-1234-123456789012/cancel"
```

## Response Format

Processed chunks are returned in the following format:

```json
{
  "status": "completed",
  "result": [
    {
      "id": "chunk_001",
      "metadata": {
        "page_num_int": [1, 2],
        "original_filename": "document.pdf",
        "chunk_size": 1024,
        "chunk_overlap": 100
      },
      "text": "Chunk content text...",
      "embedding": [0.1, 0.2, 0.3, ...]  // if with_embeddings=true
    }
  ],
  "error": null
}
```

## Supported File Formats

- **Documents**: PDF, DOCX, PPTX
- **Text**: HTML, TXT
- **Images**: JPG, JPEG, PNG (with OCR)
- **URLs**: HTTP/HTTPS links, Google Drive links

## Performance Considerations

- **Large Documents**: The service handles large documents by processing in batches
- **Memory Management**: Uses multiprocessing to isolate memory-intensive operations
- **Timeout Handling**: Jobs timeout after 3 minutes by default (configurable)
- **Concurrency**: Uses FastAPI's async capabilities and background tasks

## Error Handling

The service provides comprehensive error handling:

- **Validation Errors**: Proper HTTP status codes and error messages
- **Processing Failures**: Graceful fallback to basic text extraction
- **Network Issues**: Retry logic for external service calls
- **File Errors**: Support for corrupted or malformed files

## Monitoring

- **Health Endpoints**: `/health` and `/liveness` for monitoring
- **Logging**: Comprehensive logging for debugging and monitoring
- **Metrics**: Built-in performance tracking (can be extended with Prometheus)

## Development

### Project Structure

```
app/
├── main.py              # FastAPI application and endpoints
├── models.py           # Pydantic models and data structures
├── processing.py       # Document processing logic
├── embeddings.py       # Embedding service integration
├── config.py          # Configuration settings
└── jivas_embeddings.py # Jivas embeddings client
```

### Adding New Features

1. **New File Formats**: Add support in `processing.py`
2. **New Embedding Services**: Implement in `embeddings.py`
3. **API Endpoints**: Add to `main.py` with proper validation
4. **Processing Logic**: Extend `DocumentProcessor` class

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs` when the service is running
- Review the health endpoints for service status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Note**: This service requires external embedding services to be configured for embedding generation. Without them, text chunking will still work, but embeddings will not be generated.
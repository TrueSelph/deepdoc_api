import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from app.embeddings import embedding_client
from app.models import ChunkMetadata, ChunkResult

logger = logging.getLogger(__name__)

# Reduce thread contention that can contribute to stalls with some backends
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Set environment variables to prevent MPS fork issues on macOS
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")


class DocumentProcessor:
    """
    Document processor that uses Docling's DocumentConverter and HybridChunker
    to produce coherent chunks with contextualization.
    """

    def __init__(self):
        self.supported_formats = [
            ".pdf",
            ".docx",
            ".pptx",
            ".html",
            ".txt",
            ".jpg",
            ".jpeg",
            ".png",
        ]
        # Thread pool for embedding generation
        self.embedding_executor = ThreadPoolExecutor(max_workers=3)

    def process_document(self, file_path: str, params: dict) -> List[ChunkResult]:
        """Process a single document using Docling's DocumentConverter and HybridChunker"""
        original_filename = self._resolve_original_filename(file_path, params)

        # On macOS, use a direct approach to avoid fork issues with MPS
        if sys.platform == "darwin":
            logger.info("Using direct processing on macOS to avoid MPS fork issues")
            try:
                return self._process_directly(file_path, original_filename, params)
            except Exception as e:
                logger.error(
                    f"Direct processing failed for {file_path}: {e}. Falling back."
                )
                fallback_processor = FallbackDocumentProcessor()
                return fallback_processor.process_document(
                    file_path, params, original_filename
                )

        # On other platforms, use the multiprocessing approach with timeout
        timeout_seconds = max(10, int(params.get("timeout_seconds", 180)))

        try:
            chunks = self._run_docling_with_timeout(
                file_path=file_path,
                original_filename=original_filename,
                timeout_seconds=timeout_seconds,
            )
            if not chunks:
                raise RuntimeError("Docling worker returned no chunks")

            # Generate embeddings if requested
            if params.get("with_embeddings", False) and chunks:
                self._attach_embeddings(chunks)

            return chunks

        except Exception as e:
            logger.error(
                f"Docling processing failed for {file_path}: {e}. Falling back."
            )
            fallback_processor = FallbackDocumentProcessor()
            return fallback_processor.process_document(
                file_path, params, original_filename
            )

    def _process_directly(
        self, file_path: str, original_filename: str, params: dict
    ) -> List[ChunkResult]:
        """Process document directly without multiprocessing (for macOS)"""
        try:
            from docling.chunking import HybridChunker
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(f"Required Docling components not available: {e}")

        # Convert the document
        converter = DocumentConverter()
        result = converter.convert(source=file_path)
        doc = result.document

        # Create chunker and generate chunks
        chunker = HybridChunker()
        chunk_iter = chunker.chunk(dl_doc=doc)

        chunks = []
        for i, chunk in enumerate(chunk_iter):
            # Use contextualized text for better results
            enriched_text = chunker.contextualize(chunk=chunk)
            text = (enriched_text or "").strip()

            if not text:
                # Fallback to regular text if contextualization returns empty
                text = (chunk.text or "").strip()
                if not text:
                    continue

            # Extract page information
            pages = self._collect_chunk_pages(chunk)

            chunk_id = f"chunk_{i+1:03d}"
            metadata = ChunkMetadata(
                page_num_int=pages,
                original_filename=original_filename,
                chunk_size=len(text),
                chunk_overlap=0,
            )

            chunks.append(ChunkResult(id=chunk_id, metadata=metadata, text=text))

        # Generate embeddings if requested
        if params.get("with_embeddings", False) and chunks:
            self._attach_embeddings(chunks)

        return chunks

    def _collect_chunk_pages(self, dl_chunk) -> List[int]:
        """Collect 1-based page numbers from a docling chunk."""
        pages: Set[int] = set()

        # Try different attributes that might contain page information
        for attr in ("pages", "page_numbers", "page_nums"):
            val = getattr(dl_chunk, attr, None)
            if val:
                try:
                    if isinstance(val, (list, tuple, set)):
                        for p in val:
                            try:
                                p_int = int(p)
                                pages.add(p_int if p_int >= 1 else p_int + 1)
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(val, int):
                        pages.add(val if val >= 1 else val + 1)
                except Exception:
                    pass

        # Check for source_spans or spans with page indices
        spans = getattr(dl_chunk, "source_spans", None) or getattr(
            dl_chunk, "spans", None
        )
        if spans:
            for sp in spans:
                for attr in ("page", "page_idx", "page_number", "page_num"):
                    page_idx = getattr(sp, attr, None)
                    if page_idx is not None:
                        try:
                            p_int = int(page_idx)
                            pages.add(p_int + 1)  # assume 0-based in spans
                            break
                        except (ValueError, TypeError):
                            pass

        # Check metadata for page information
        meta = getattr(dl_chunk, "metadata", None) or getattr(dl_chunk, "meta", None)
        if meta is not None:
            for attr in ("pages", "page_numbers", "page_nums"):
                meta_pages = getattr(meta, attr, None)
                if meta_pages:
                    try:
                        if isinstance(meta_pages, (list, tuple, set)):
                            for p in meta_pages:
                                try:
                                    p_int = int(p)
                                    pages.add(p_int if p_int >= 1 else p_int + 1)
                                except (ValueError, TypeError):
                                    pass
                        elif isinstance(meta_pages, int):
                            pages.add(meta_pages if meta_pages >= 1 else meta_pages + 1)
                    except Exception:
                        pass

        if not pages:
            pages = {1}

        return sorted(pages)

    def _run_docling_with_timeout(
        self,
        file_path: str,
        original_filename: str,
        timeout_seconds: int,
    ) -> List[ChunkResult]:
        """
        Spawn a worker process to run Docling's DocumentConverter and HybridChunker.
        Returns a list of ChunkResult or raises on failure/timeout.
        """
        # Use 'spawn' method to avoid fork issues
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_docling_worker_process,
            args=(file_path, child_conn),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # close child end in parent

        try:
            if parent_conn.poll(timeout_seconds):
                result = parent_conn.recv()
            else:
                # Timeout: terminate and raise
                proc.terminate()
                proc.join(timeout=5)
                raise TimeoutError(
                    f"Docling processing timed out after {timeout_seconds}s"
                )
        finally:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
            try:
                parent_conn.close()
            except Exception:
                pass

        if not isinstance(result, dict):
            raise RuntimeError("Invalid result type from Docling worker")

        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Unknown Docling worker error"))

        # Convert worker chunk dicts to ChunkResult with metadata
        chunks_dicts: List[Dict[str, Any]] = result.get("chunks", [])
        chunks: List[ChunkResult] = []
        for idx, ch in enumerate(chunks_dicts, start=1):
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            pages = ch.get("pages") or [1]
            chunk_id = f"chunk_{idx:03d}"
            meta = ChunkMetadata(
                page_num_int=pages,
                original_filename=original_filename,
                chunk_size=len(text),
                chunk_overlap=0,
            )
            chunks.append(ChunkResult(id=chunk_id, metadata=meta, text=text))

        return chunks

    def _resolve_original_filename(self, file_path: str, params: dict) -> str:
        """Resolve original filename from params; strip UUID prefix if present."""
        original_filenames = params.get("original_filenames", {}) or {}
        original_filename = original_filenames.get(
            file_path, os.path.basename(file_path)
        )

        # Remove job ID prefix if looks like a UUID_ prefix
        if original_filename and "_" in original_filename:
            parts = original_filename.split("_", 1)
            if len(parts) > 1 and len(parts[0]) == 36:
                original_filename = parts[1]
        return original_filename

    def _attach_embeddings(self, chunks: List[ChunkResult]) -> None:
        """Generate embeddings for chunks one at a time to handle large documents"""
        if not chunks:
            return

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        successful_embeddings = 0
        failed_embeddings = 0

        for i, chunk in enumerate(chunks):
            try:
                # Process one chunk at a time
                embedding = embedding_client.generate_embedding(chunk.text)
                if embedding:
                    chunk.embeddings = embedding
                    successful_embeddings += 1
                else:
                    logger.warning(f"Failed to generate embedding for chunk {i+1}")
                    failed_embeddings += 1

                # Log progress every 10 chunks
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(chunks)} chunks for embeddings")

            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i+1}: {e}")
                failed_embeddings += 1
                # Continue with next chunk instead of failing completely

        logger.info(
            f"Embedding generation completed: {successful_embeddings} successful, {failed_embeddings} failed"
        )

        if failed_embeddings > 0:
            logger.warning(f"{failed_embeddings} chunks failed to generate embeddings")


def _docling_worker_process(
    file_path: str,
    conn,
) -> None:
    """
    Worker process entrypoint: uses DocumentConverter and HybridChunker
    with contextualization for optimal chunking.
    """
    # Set environment variables to prevent MPS issues
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")

    try:
        # Try to import the required components
        try:
            from docling.chunking import HybridChunker
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(f"Required Docling components not available: {e}")

        # Convert the document
        converter = DocumentConverter()
        result = converter.convert(source=file_path)
        doc = result.document

        # Create chunker and generate chunks
        chunker = HybridChunker()
        chunk_iter = chunker.chunk(dl_doc=doc)

        out_chunks: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunk_iter):
            # Use contextualized text for better results
            enriched_text = chunker.contextualize(chunk=chunk)
            text = (enriched_text or "").strip()

            if not text:
                # Fallback to regular text if contextualization returns empty
                text = (chunk.text or "").strip()
                if not text:
                    continue

            # Extract page information
            pages = _collect_chunk_pages(chunk)

            out_chunks.append({"text": text, "pages": pages, "chunk_index": i})

        conn.send({"ok": True, "chunks": out_chunks})
        conn.close()

    except Exception as e:
        try:
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
            conn.close()
        except Exception:
            pass


def _collect_chunk_pages(dl_chunk) -> List[int]:
    """Collect 1-based page numbers from a docling chunk."""
    pages: Set[int] = set()

    # Try different attributes that might contain page information
    for attr in ("pages", "page_numbers", "page_nums"):
        val = getattr(dl_chunk, attr, None)
        if val:
            try:
                if isinstance(val, (list, tuple, set)):
                    for p in val:
                        try:
                            p_int = int(p)
                            pages.add(p_int if p_int >= 1 else p_int + 1)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(val, int):
                    pages.add(val if val >= 1 else val + 1)
            except Exception:
                pass

    # Check for source_spans or spans with page indices
    spans = getattr(dl_chunk, "source_spans", None) or getattr(dl_chunk, "spans", None)
    if spans:
        for sp in spans:
            for attr in ("page", "page_idx", "page_number", "page_num"):
                page_idx = getattr(sp, attr, None)
                if page_idx is not None:
                    try:
                        p_int = int(page_idx)
                        pages.add(p_int + 1)  # assume 0-based in spans
                        break
                    except (ValueError, TypeError):
                        pass

    # Check metadata for page information
    meta = getattr(dl_chunk, "metadata", None) or getattr(dl_chunk, "meta", None)
    if meta is not None:
        for attr in ("pages", "page_numbers", "page_nums"):
            meta_pages = getattr(meta, attr, None)
            if meta_pages:
                try:
                    if isinstance(meta_pages, (list, tuple, set)):
                        for p in meta_pages:
                            try:
                                p_int = int(p)
                                pages.add(p_int if p_int >= 1 else p_int + 1)
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(meta_pages, int):
                        pages.add(meta_pages if meta_pages >= 1 else meta_pages + 1)
                except Exception:
                    pass

    if not pages:
        pages = {1}

    return sorted(pages)


# Fallback processor remains the same as in your original code
class FallbackDocumentProcessor:
    """Fallback processor if Docling is not available or times out."""

    def __init__(self):
        self.supported_formats = [
            ".pdf",
            ".docx",
            ".pptx",
            ".html",
            ".txt",
            ".jpg",
            ".jpeg",
            ".png",
        ]

    def process_document(
        self, file_path: str, params: dict, original_filename: Optional[str] = None
    ) -> List[ChunkResult]:
        try:
            if original_filename is None:
                original_filename = os.path.basename(file_path)
                if "_" in original_filename:
                    parts = original_filename.split("_", 1)
                    if len(parts) > 1 and len(parts[0]) == 36:
                        original_filename = parts[1]

            file_ext = os.path.splitext(file_path)[1].lower()
            content = ""
            if file_ext == ".pdf":
                content = self._extract_text_from_pdf(file_path)
            elif file_ext == ".docx":
                content = self._extract_text_from_docx(file_path)
            elif file_ext in [".txt", ".html"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif file_ext in [".jpg", ".jpeg", ".png"]:
                content = self._extract_text_from_image(file_path)
            else:
                content = f"Content from {original_filename}"

            page_info = {"original_filename": original_filename, "pages": [1]}
            chunks = self._chunk_content(content, original_filename, page_info)

            # Generate embeddings if requested
            if params.get("with_embeddings", False) and chunks:
                self._attach_embeddings(chunks)

            return chunks

        except Exception as e:
            logger.error(f"Fallback processing failed for {file_path}: {e}")
            return [
                ChunkResult(
                    id="chunk_001",
                    metadata=ChunkMetadata(
                        page_num_int=[1],
                        original_filename=original_filename
                        or os.path.basename(file_path),
                    ),
                    text=f"Error processing document: {str(e)}",
                )
            ]

    def _attach_embeddings(self, chunks: List[ChunkResult]) -> None:
        """Generate embeddings for chunks using the external embedding service"""
        try:
            texts = [c.text for c in chunks]

            # Use the embedding client to generate embeddings
            embeddings = embedding_client.generate_embeddings(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Continue without embeddings rather than failing the whole job

    def _extract_text_from_pdf(self, file_path: str) -> str:
        try:
            import PyPDF2

            text = ""
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            logger.warning("PyPDF2 not available for PDF extraction")
            return f"PDF content would be extracted here: {os.path.basename(file_path)}"

    def _extract_text_from_docx(self, file_path: str) -> str:
        try:
            import docx

            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            logger.warning("python-docx not available for DOCX extraction")
            return (
                f"DOCX content would be extracted here: {os.path.basename(file_path)}"
            )

    def _extract_text_from_image(self, file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            logger.warning(
                "Tesseract OCR not available. Install pytesseract and tesseract-ocr"
            )
            return f"Image OCR content would be extracted here: {os.path.basename(file_path)}"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return f"OCR extraction failed for: {os.path.basename(file_path)}"

    def _chunk_content(
        self,
        content: str,
        original_filename: str,
        page_info: dict,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> List[ChunkResult]:
        chunks = []
        start = 0
        chunk_count = 0
        n = len(content or "")

        while start < n:
            chunk_count += 1
            end = min(start + chunk_size, n)

            if end < n:
                break_pos = content.rfind(" ", start, end)
                if break_pos != -1 and break_pos > start + chunk_size // 2:
                    end = break_pos + 1

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunk_id = f"chunk_{chunk_count:03d}"
                metadata = ChunkMetadata(
                    page_num_int=page_info.get("pages", [1]),
                    original_filename=original_filename,
                    chunk_size=len(chunk_text),
                    chunk_overlap=overlap if start > 0 else 0,
                )
                chunks.append(
                    ChunkResult(id=chunk_id, metadata=metadata, text=chunk_text)
                )

            new_start = end - overlap
            start = end if new_start <= start else new_start
            if start <= 0:
                start = end

        return chunks


# Create processor instance
document_processor = DocumentProcessor()

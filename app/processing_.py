"""Document processing module with Docling integration."""

import contextlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from app.embeddings import embedding_client
from app.models import ChunkMetadata, ChunkResult

logger = logging.getLogger(__name__)

# Reduce thread contention that can contribute to stalls with some backends
os.environ.update(
    {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
)

# Set environment variables to prevent MPS fork issues on macOS
if sys.platform == "darwin":
    os.environ.update({"PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTORCH_MPS_DISABLE": "1"})

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HTML_EXTS = {".html", ".htm", ".xhtml"}
TEXT_EXTS = {".txt", ".md", ".rst", ".log", ".xml"}
OFFICE_LIKE_EXTS = {
    ".doc",
    ".docx",
    ".rtf",
    ".xls",
    ".xlsx",
    ".csv",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".pages",
    ".numbers",
    ".key",
}


def _which(cmd: str) -> Optional[str]:
    """Shallow wrapper around shutil.which with typing."""
    return shutil.which(cmd)


def _ensure_dir(p: Union[str, Path]) -> None:
    """Ensure parent directory of path exists."""
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _extract_original_filename(file_path: str, params: dict) -> str:
    """Extract original filename from parameters, stripping UUID prefix if present.

    Args:
        file_path: Path to the document file
        params: Processing parameters dictionary

    Returns:
        Original filename without job ID prefix
    """
    original_filenames = params.get("original_filenames", {}) or {}
    original_filename = original_filenames.get(file_path, os.path.basename(file_path))

    # Remove job ID prefix if looks like a UUID_ prefix
    if "_" in original_filename:
        parts = original_filename.split("_", 1)
        if len(parts) > 1 and len(parts[0]) == 36:
            original_filename = parts[1]

    return original_filename


class DocumentConverter:
    """Handles document conversion to PDF for various file formats."""

    def __init__(self) -> None:
        """Initialize the document converter."""
        self._soffice_path = _which("soffice") or _which("libreoffice")

    def convert_to_pdf(
        self, input_path: str, output_path: str, job_id: Optional[str] = None
    ) -> bool:
        """Convert document to PDF using appropriate method based on file extension.

        Args:
            input_path: Path to input file
            output_path: Path to output PDF file
            job_id: Job ID for progress tracking (optional)

        Returns:
            True if conversion succeeded, False otherwise
        """
        file_ext = os.path.splitext(input_path)[1].lower()

        # Send progress update for conversion start
        if job_id:
            from app.main import update_job_progress

            update_job_progress(
                job_id,
                f"Converting {os.path.basename(input_path)} to PDF",
                5,
                {
                    "current_operation": "Document conversion",
                    "conversion_format": file_ext,
                    "stage": "conversion_start",
                },
            )

        # Already PDF: copy if different path
        if file_ext == ".pdf":
            if os.path.abspath(input_path) == os.path.abspath(output_path):
                return True
            _ensure_dir(output_path)
            try:
                shutil.copy2(input_path, output_path)
                # Send progress update for completion
                if job_id:
                    update_job_progress(
                        job_id,
                        "PDF copy completed",
                        15,
                        {
                            "current_operation": "Document conversion",
                            "stage": "conversion_complete",
                            "method": "copy",
                        },
                    )
                return True
            except Exception as e:
                logger.exception("Failed to copy PDF: %s", e)
                return False

        # Images - Pillow
        if file_ext in IMAGE_EXTS:
            return self._convert_image_to_pdf(input_path, output_path)

        # HTML - try pdfkit first then WeasyPrint
        if file_ext in HTML_EXTS:
            if self._convert_with_pdfkit(input_path, output_path):
                return True
            logger.warning("pdfkit failed; trying WeasyPrint for HTML.")
            return self._convert_with_weasyprint(input_path, output_path)

        # Plain text / Markdown / reST / logs / XML
        if file_ext in TEXT_EXTS:
            return self._convert_text_to_pdf(input_path, output_path)

        # Office-like docs - LibreOffice first
        if file_ext in OFFICE_LIKE_EXTS:
            if self._convert_with_libreoffice(input_path, output_path):
                return True
            logger.warning(
                "LibreOffice failed for %s; no secondary converter available.", file_ext
            )
            return False

        # Unknown: try LibreOffice; if it fails, try WeasyPrint as a last resort
        if self._convert_with_libreoffice(input_path, output_path):
            return True
        logger.warning("Unknown format; attempting WeasyPrint as a last resort.")
        return self._convert_with_weasyprint(input_path, output_path)

    def _convert_with_pdfkit(self, input_path: str, output_path: str) -> bool:
        """Convert HTML to PDF using pdfkit/wkhtmltopdf."""
        try:
            import pdfkit  # type: ignore
        except ImportError:
            logger.debug("pdfkit not available.")
            return False

        _ensure_dir(output_path)
        try:
            # Allow overriding wkhtmltopdf binary via env var if needed
            wkhtml_path = os.getenv("WKHTMLTOPDF_PATH")
            config = (
                pdfkit.configuration(wkhtmltopdf=wkhtml_path) if wkhtml_path else None
            )
            options = {
                "quiet": "",
                "enable-local-file-access": None,
                "load-error-handling": "ignore",
                "load-media-error-handling": "ignore",
                "print-media-type": None,
                "disable-smart-shrinking": None,
                "margin-top": "10mm",
                "margin-right": "10mm",
                "margin-bottom": "10mm",
                "margin-left": "10mm",
            }
            pdfkit.from_file(
                input_path, output_path, options=options, configuration=config
            )
            return os.path.exists(output_path)
        except Exception as e:
            logger.warning("pdfkit conversion failed: %s", e)
            return False

    def _convert_with_weasyprint(self, input_path: str, output_path: str) -> bool:
        """Convert HTML to PDF using WeasyPrint."""
        try:
            from weasyprint import CSS, HTML  # type: ignore
        except ImportError:
            logger.debug("WeasyPrint not available.")
            return False

        _ensure_dir(output_path)
        try:
            base_url = os.path.dirname(os.path.abspath(input_path))
            # Minimal default CSS to avoid overly cramped output
            default_css = CSS(
                string="""
                @page { size: A4; margin: 10mm; }
                body { font-family: sans-serif; font-size: 12px; }
                pre { white-space: pre-wrap; word-wrap: break-word; }
                img { max-width: 100%; }
            """
            )
            HTML(filename=input_path, base_url=base_url).write_pdf(
                output_path, stylesheets=[default_css]
            )
            return os.path.exists(output_path)
        except Exception as e:
            logger.warning("WeasyPrint conversion failed: %s", e)
            return False

    def _convert_text_to_pdf(self, input_path: str, output_path: str) -> bool:
        """Convert plaintext/Markdown/reST/XML to PDF via HTML rendering."""
        try:
            with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except Exception as e:
            logger.exception("Failed reading text file: %s", e)
            return False

        ext = os.path.splitext(input_path)[1].lower()
        html_body: str

        # Try Markdown
        if ext == ".md":
            try:
                import markdown  # type: ignore

                html_body = markdown.markdown(
                    raw, extensions=["extra", "tables", "sane_lists", "toc"]
                )
            except Exception as e:
                logger.debug(
                    "Markdown conversion failed (%s); falling back to <pre>.", e
                )
                html_body = f"<pre>{raw}</pre>"

        # Try reStructuredText
        elif ext == ".rst":
            try:
                from docutils.core import publish_parts  # type: ignore

                parts = publish_parts(source=raw, writer_name="html5")
                html_body = parts.get("html_body", "") or f"<pre>{raw}</pre>"
            except Exception as e:
                logger.debug("reST conversion failed (%s); falling back to <pre>.", e)
                html_body = f"<pre>{raw}</pre>"
        else:
            # Plain text, logs, xml
            if ext == ".xml":
                html_body = raw
            else:
                html_body = f"<pre>{raw}</pre>"

        html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{os.path.basename(input_path)}</title>
<style>
body {{ font-family: sans-serif; font-size: 12px; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; }}
code {{ white-space: pre-wrap; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
        # Write to a temp HTML next to output and convert
        tmp_html = Path(output_path).with_suffix(".tmp.html")
        try:
            tmp_html.write_text(html_doc, encoding="utf-8")
        except Exception as e:
            logger.exception("Failed writing temp HTML: %s", e)
            return False

        try:
            # Prefer WeasyPrint for internal, fallback to pdfkit
            if self._convert_with_weasyprint(str(tmp_html), output_path):
                return True
            logger.warning(
                "WeasyPrint failed; trying pdfkit for text-based conversion."
            )
            return self._convert_with_pdfkit(str(tmp_html), output_path)
        finally:
            with contextlib.suppress(Exception):
                tmp_html.unlink()

    def _convert_with_libreoffice(self, input_path: str, output_path: str) -> bool:
        """Convert Office-like documents to PDF via headless LibreOffice."""
        if not self._soffice_path:
            logger.debug("LibreOffice (soffice) not found in PATH.")
            return False

        in_path = Path(input_path).resolve()
        out_path = Path(output_path).resolve()
        _ensure_dir(out_path)

        # LibreOffice writes to --outdir with same stem + .pdf
        tmp_outdir = out_path.parent / f".lo_{in_path.stem}"
        tmp_outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._soffice_path,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_outdir),
            str(in_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            candidate = tmp_outdir / (in_path.stem + ".pdf")
            if proc.returncode != 0 or not candidate.exists():
                logger.warning(
                    "LibreOffice conversion failed. rc=%s, stderr=%s",
                    proc.returncode,
                    proc.stderr,
                )
                return False
            # Move to desired output path
            if out_path.exists():
                with contextlib.suppress(Exception):
                    out_path.unlink()
            candidate.replace(out_path)
            return True
        except Exception:
            logger.exception("LibreOffice conversion error")
            return False
        finally:
            # Best-effort cleanup
            with contextlib.suppress(Exception):
                for p in tmp_outdir.iterdir():
                    with contextlib.suppress(Exception):
                        p.unlink()
                tmp_outdir.rmdir()

    def _convert_image_to_pdf(self, input_path: str, output_path: str) -> bool:
        """Convert image (including multi-frame TIFF) to PDF via Pillow."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            logger.debug("Pillow not installed.")
            return False

        in_path = Path(input_path).resolve()
        out_path = Path(output_path).resolve()
        _ensure_dir(out_path)

        try:
            with Image.open(in_path) as im:
                frames = []
                try:
                    i = 0
                    while True:
                        im.seek(i)
                        frames.append(im.convert("RGB"))
                        i += 1
                except EOFError:
                    pass

                if len(frames) == 1:
                    frames[0].save(out_path, "PDF", resolution=300.0)
                else:
                    frames[0].save(
                        out_path,
                        "PDF",
                        save_all=True,
                        append_images=frames[1:],
                        resolution=300.0,
                    )
            return out_path.exists()
        except Exception:
            logger.exception("Image to PDF failed")
            return False


class DocumentProcessor:
    """Document processor using Docling's DocumentConverter and HybridChunker."""

    def __init__(self) -> None:
        """Initialize the DocumentProcessor with supported formats and thread pool."""
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
        self.document_converter = DocumentConverter()

    def process_document(self, file_path: str, params: dict) -> List[ChunkResult]:
        """Process a single document using Docling's DocumentConverter and HybridChunker.

        Args:
            file_path: Path to the document file
            params: Processing parameters dictionary

        Returns:
            List of processed chunks with metadata
        """
        original_filename = _extract_original_filename(file_path, params)
        job_id = params.get(
            "job_id"
        )  # Extract job_id from params for progress tracking
        logger.info("Starting document processing for: %s", original_filename)

        # Log milestone for document processing start
        if job_id:
            from app.main import log_progress_milestone

            log_progress_milestone(
                job_id, f"Document processing started for {original_filename}", 10
            )

        # use the multiprocessing approach with timeout
        job_timeout = os.getenv("JOB_TIMEOUT", "14400")
        timeout_seconds = max(10, int(params.get("timeout_seconds", job_timeout)))

        try:
            chunks = self._run_docling_with_timeout(
                file_path=file_path,
                original_filename=original_filename,
                timeout_seconds=timeout_seconds,
                job_id=job_id,
            )
            if not chunks:
                raise RuntimeError("Docling worker returned no chunks")

            # Generate embeddings if requested
            if params.get("with_embeddings", False) and chunks:
                if job_id:
                    from app.main import log_progress_milestone

                    log_progress_milestone(job_id, "Starting embedding generation", 75)
                self._attach_embeddings(chunks, job_id)

            logger.info("Successfully processed document into %d chunks", len(chunks))

            # Log successful completion milestone
            if job_id:
                from app.main import log_progress_milestone

                log_progress_milestone(
                    job_id, f"Document processing completed: {len(chunks)} chunks", 90
                )

            return chunks

        except Exception as e:
            logger.exception(
                "Docling processing failed for %s: %s. Falling back.", file_path, e
            )

            # Update progress to indicate fallback mode
            if job_id:
                from app.main import update_job_progress

                update_job_progress(
                    job_id,
                    "Primary processing failed, using fallback method",
                    10,
                    {
                        "current_operation": "Fallback processing",
                        "fallback_reason": str(type(e).__name__),
                        "original_error": str(e),
                    },
                )

            fallback_processor = FallbackDocumentProcessor()
            return fallback_processor.process_document(
                file_path, params, original_filename
            )

    def _process_with_docling(
        self, file_path: str, conn: Connection, job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Common function to process document using Docling with PDF conversion.

        This function ensures documents are converted to PDF format before processing.

        Args:
            file_path: Path to the document file

        Returns:
            List of chunk data dictionaries with text, pages, and optional bbox

        Raises:
            ImportError: If required Docling components not available
            RuntimeError: If document conversion or processing fails
        """
        try:
            from docling.chunking import HybridChunker
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(f"Required Docling components not available: {e}")

        # Get upload directory and create temp subfolder
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        temp_dir = os.path.join(upload_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        temp_pdf_path: Optional[str] = None

        try:
            # First, detect the input format
            file_ext = os.path.splitext(file_path)[1].lower()
            is_pdf = file_ext == ".pdf"
            logger.info(f"Processing file with extension: {file_ext}, is_pdf: {is_pdf}")

            # Convert non-PDF files to PDF before processing
            if not is_pdf:
                # Create temporary PDF file in the temp subfolder
                temp_filename = f"converted_{uuid.uuid4().hex}.pdf"
                temp_pdf_path = os.path.join(temp_dir, temp_filename)
                logger.info(f"Converting {file_path} to PDF: {temp_pdf_path}")

                conversion_success = self.document_converter.convert_to_pdf(
                    file_path, temp_pdf_path, job_id
                )
                logger.info(f"PDF conversion result: {conversion_success}")

                if conversion_success:
                    processing_file_path = temp_pdf_path
                    logger.info(f"Using converted PDF: {processing_file_path}")
                else:
                    # If conversion fails, try to process the original file directly
                    logger.warning(
                        "PDF conversion failed for %s, attempting direct processing",
                        file_path,
                    )
                    processing_file_path = file_path

                original_format = file_ext

            else:
                # File is already PDF, use directly
                processing_file_path = file_path
                original_format = ".pdf"
                logger.info(f"Processing PDF directly: {processing_file_path}")

            # Verify the processing file exists
            if not os.path.exists(processing_file_path):
                raise FileNotFoundError(
                    f"Processing file does not exist: {processing_file_path}"
                )

            # Send progress update: conversion complete
            try:
                conn.send(
                    {"progress": {"stage": "conversion_complete", "progress": 10}}
                )
                logger.info("Sent conversion_complete progress update")
            except Exception as progress_e:
                logger.warning(f"Failed to send conversion progress: {progress_e}")

            # Process the document with Docling
            logger.info("Starting Docling document conversion...")
            try:
                converter = DocumentConverter()
                result = converter.convert(source=processing_file_path)
                doc = result.document
                logger.info("Docling conversion successful")
            except Exception as docling_e:
                logger.error(
                    f"Docling conversion failed: {type(docling_e).__name__}: {docling_e}"
                )
                # Try to provide more specific error information
                if "CUDA" in str(docling_e) or "GPU" in str(docling_e):
                    raise RuntimeError(f"Docling GPU/CUDA error: {docling_e}")
                elif "memory" in str(docling_e).lower():
                    raise RuntimeError(f"Docling memory error: {docling_e}")
                elif "file" in str(docling_e).lower():
                    raise RuntimeError(f"Docling file processing error: {docling_e}")
                else:
                    raise RuntimeError(f"Docling processing error: {docling_e}")

            # Get document statistics for progress tracking
            doc_stats = self._get_document_statistics(doc)

            # Send progress update: document loaded
            try:
                conn.send(
                    {
                        "progress": {
                            "stage": "document_loaded",
                            "progress": 20,
                            "stats": doc_stats,
                        }
                    }
                )
                logger.info(
                    f"Sent document_loaded progress update with stats: {doc_stats}"
                )
            except Exception as progress_e:
                logger.warning(f"Failed to send document_loaded progress: {progress_e}")

            # Create chunker and generate chunks with progress tracking
            chunker = HybridChunker()
            chunk_iter = chunker.chunk(dl_doc=doc)

            chunks_data: List[Dict[str, Any]] = []
            processed_elements = 0
            total_elements = 0

            # First pass: count total elements if not available from stats
            if doc_stats.get("elements", 0) == 0:
                temp_chunks = list(chunk_iter)
                total_elements = len(temp_chunks)
                chunk_iter = iter(temp_chunks)  # Reset iterator
            else:
                total_elements = doc_stats.get("elements", 0)

            for chunk in chunk_iter:
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

                # Extract bounding box data
                bbox_data = self._collect_bounding_box(chunk)

                chunk_data = {
                    "text": text,
                    "pages": pages,
                    "bbox": bbox_data,
                    "original_format": original_format,
                }
                chunks_data.append(chunk_data)

                # Update progress based on actual processing
                processed_elements += 1
                if processed_elements % 3 == 0:  # Send progress more frequently
                    progress_percent = 20 + int(
                        (processed_elements / max(total_elements, 1)) * 60
                    )
                    try:
                        conn.send(
                            {
                                "progress": {
                                    "stage": "chunking",
                                    "progress": progress_percent,
                                    "chunks_processed": processed_elements,
                                    "total_chunks": total_elements,
                                }
                            }
                        )
                        logger.debug(
                            f"Sent chunking progress: {processed_elements}/{total_elements} ({progress_percent}%)"
                        )
                    except Exception as progress_e:
                        logger.warning(
                            f"Failed to send chunking progress: {progress_e}"
                        )

            # Send final progress update
            with contextlib.suppress(Exception):
                conn.send(
                    {
                        "progress": {
                            "stage": "processing_complete",
                            "progress": 80,
                            "final_stats": {
                                "total_chunks": len(chunks_data),
                                "total_pages": doc_stats.get("pages", 0),
                                "file_size_bytes": doc_stats.get("file_size", 0),
                            },
                        }
                    }
                )

            return chunks_data

        except Exception as e:
            # Log the full exception details
            logger.error(f"Document processing failed: {type(e).__name__}: {e}")
            logger.exception("Full exception traceback:")

            # Send error progress update with more details
            try:
                conn.send(
                    {
                        "progress": {
                            "stage": "error",
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "file_path": file_path,
                            "processing_stage": "document_processing",
                        }
                    }
                )
                logger.info("Sent error progress update to parent process")
            except Exception as progress_e:
                logger.error(f"Failed to send error progress: {progress_e}")

            # Re-raise with more context
            raise RuntimeError(f"Document processing failed: {type(e).__name__}: {e}")

        finally:
            # Clean up temporary PDF file if it was created
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                with contextlib.suppress(OSError):
                    os.unlink(temp_pdf_path)

    def _collect_chunk_pages(self, dl_chunk: object) -> List[int]:
        """Collect 1-based page numbers from a docling chunk.

        Args:
            dl_chunk: Docling chunk object

        Returns:
            Sorted list of page numbers
        """
        pages: Set[int] = set()

        # First, try to extract from metadata provenance (most reliable)
        meta = getattr(dl_chunk, "metadata", None) or getattr(dl_chunk, "meta", None)
        if meta is not None:
            # Check for provenance information in meta
            prov = getattr(meta, "prov", None)
            if prov and hasattr(prov, "__iter__"):
                for provenance_item in prov:
                    page_no = getattr(provenance_item, "page_no", None)
                    if page_no is not None:
                        try:
                            p_int = int(page_no)
                            # Page numbers are typically 1-based
                            pages.add(p_int if p_int >= 1 else p_int + 1)
                        except (ValueError, TypeError):
                            pass

            # Also check for direct page_no attribute in meta
            page_no = getattr(meta, "page_no", None)
            if page_no is not None:
                try:
                    p_int = int(page_no)
                    pages.add(p_int if p_int >= 1 else p_int + 1)
                except (ValueError, TypeError):
                    pass

        # Try different attributes that might contain page information
        for attr in ("pages", "page_numbers", "page_nums", "page_no"):
            val = getattr(dl_chunk, attr, None)
            if val:
                with contextlib.suppress(Exception):
                    if isinstance(val, (list, tuple, set)):
                        for p in val:
                            try:
                                p_int = int(p)
                                pages.add(p_int if p_int >= 1 else p_int + 1)
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(val, int):
                        pages.add(val if val >= 1 else val + 1)

        # Check for source_spans or spans with page indices
        spans = getattr(dl_chunk, "source_spans", None) or getattr(
            dl_chunk, "spans", None
        )
        if spans:
            for sp in spans:
                for attr in ("page", "page_idx", "page_number", "page_num", "page_no"):
                    page_idx = getattr(sp, attr, None)
                    if page_idx is not None:
                        try:
                            p_int = int(page_idx)
                            # Assume 0-based in spans, convert to 1-based
                            pages.add(p_int + 1 if p_int >= 0 else 1)
                            break
                        except (ValueError, TypeError):
                            pass

        # Additional check: look for page information in doc_items if present
        if meta is not None:
            doc_items = getattr(meta, "doc_items", None)
            if doc_items and hasattr(doc_items, "__iter__"):
                for doc_item in doc_items:
                    prov = getattr(doc_item, "prov", None)
                    if prov and hasattr(prov, "__iter__"):
                        for provenance_item in prov:
                            page_no = getattr(provenance_item, "page_no", None)
                            if page_no is not None:
                                try:
                                    p_int = int(page_no)
                                    pages.add(p_int if p_int >= 1 else p_int + 1)
                                except (ValueError, TypeError):
                                    pass

        # If no pages found, try to get from the chunk's text content or context
        if not pages:
            # Last resort: check if we can infer from the chunk's string representation
            chunk_str = str(dl_chunk)
            page_match = re.search(
                r"page[_\s]*(no|num|number)?[_\s]*[=:]\s*(\d+)",
                chunk_str,
                re.IGNORECASE,
            )
            if page_match:
                try:
                    page_no = int(page_match.group(2))
                    pages.add(page_no if page_no >= 1 else page_no + 1)
                except (ValueError, TypeError):
                    pass

        # Final fallback if no pages detected
        if not pages:
            pages = {1}

        return sorted(pages)

    def _collect_bounding_box(self, dl_chunk: object) -> Optional[Dict[str, float]]:
        """Collect bounding box data from a docling chunk.

        Args:
            dl_chunk: Docling chunk object

        Returns:
            Dictionary with bounding box coordinates or None if not available
        """
        bbox_data: Optional[Dict[str, float]] = None

        # First, try to extract from metadata provenance (most reliable)
        meta = getattr(dl_chunk, "metadata", None) or getattr(dl_chunk, "meta", None)
        if meta is not None:
            # Check for provenance information in meta
            prov = getattr(meta, "prov", None)
            if prov and hasattr(prov, "__iter__"):
                for provenance_item in prov:
                    bbox = getattr(provenance_item, "bbox", None)
                    if bbox is not None:
                        bbox_data = self._extract_bbox_from_object(bbox)
                        if bbox_data:
                            break

            # Also check for direct bbox attribute in meta
            if bbox_data is None:
                bbox = getattr(meta, "bbox", None)
                if bbox is not None:
                    bbox_data = self._extract_bbox_from_object(bbox)

        # Check for bounding box in the chunk itself
        if bbox_data is None:
            for attr in ("bbox", "bounding_box", "boundingbox", "rect", "rectangle"):
                bbox = getattr(dl_chunk, attr, None)
                if bbox is not None:
                    bbox_data = self._extract_bbox_from_object(bbox)
                    if bbox_data:
                        break

        # Check for source_spans or spans with bounding box
        if bbox_data is None:
            spans = getattr(dl_chunk, "source_spans", None) or getattr(
                dl_chunk, "spans", None
            )
            if spans:
                for sp in spans:
                    for attr in (
                        "bbox",
                        "bounding_box",
                        "boundingbox",
                        "rect",
                        "rectangle",
                    ):
                        bbox = getattr(sp, attr, None)
                        if bbox is not None:
                            bbox_data = self._extract_bbox_from_object(bbox)
                            if bbox_data:
                                break
                    if bbox_data:
                        break

        # Additional check: look for bounding box in doc_items if present
        if bbox_data is None and meta is not None:
            doc_items = getattr(meta, "doc_items", None)
            if doc_items and hasattr(doc_items, "__iter__"):
                for doc_item in doc_items:
                    prov = getattr(doc_item, "prov", None)
                    if prov and hasattr(prov, "__iter__"):
                        for provenance_item in prov:
                            bbox = getattr(provenance_item, "bbox", None)
                            if bbox is not None:
                                bbox_data = self._extract_bbox_from_object(bbox)
                                if bbox_data:
                                    break
                        if bbox_data:
                            break

        return bbox_data

    def _get_document_statistics(self, doc: object) -> Dict[str, Any]:
        """Extract statistics from docling document for progress tracking."""
        stats = {"pages": 0, "elements": 0, "file_size": 0}

        try:
            # Try to get page count
            if hasattr(doc, "pages"):
                stats["pages"] = len(doc.pages) if hasattr(doc.pages, "__len__") else 0

            # Try to get element count
            if hasattr(doc, "text_elements"):
                stats["elements"] = (
                    len(doc.text_elements)
                    if hasattr(doc.text_elements, "__len__")
                    else 0
                )
            elif hasattr(doc, "elements"):
                stats["elements"] = (
                    len(doc.elements) if hasattr(doc.elements, "__len__") else 0
                )

            # Try to get file size if source path is available
            if hasattr(doc, "source") and hasattr(doc.source, "file"):
                try:
                    import os

                    stats["file_size"] = os.path.getsize(doc.source.file)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Could not extract document statistics: {e}")

        return stats

    def _extract_bbox_from_object(self, bbox_obj: object) -> Optional[Dict[str, float]]:
        """Extract bounding box coordinates from various bbox object formats.

        Args:
            bbox_obj: Bounding box object with various attribute naming conventions

        Returns:
            Dictionary with bounding box coordinates or None if extraction fails
        """
        try:
            # Try different attribute naming patterns
            bbox_methods = [
                # l, t, r, b pattern
                lambda: {
                    "left": getattr(bbox_obj, "l", None),
                    "top": getattr(bbox_obj, "t", None),
                    "right": getattr(bbox_obj, "r", None),
                    "bottom": getattr(bbox_obj, "b", None),
                },
                # x, y, width, height pattern
                lambda: {
                    "left": getattr(bbox_obj, "x", None),
                    "top": getattr(bbox_obj, "y", None),
                    "right": (getattr(bbox_obj, "x", None) or 0)
                    + getattr(bbox_obj, "width", 0),
                    "bottom": (getattr(bbox_obj, "y", None) or 0)
                    + getattr(bbox_obj, "height", 0),
                },
                # x1, y1, x2, y2 pattern
                lambda: {
                    "left": getattr(bbox_obj, "x1", None),
                    "top": getattr(bbox_obj, "y1", None),
                    "right": getattr(bbox_obj, "x2", None),
                    "bottom": getattr(bbox_obj, "y2", None),
                },
                # left, top, right, bottom pattern
                lambda: {
                    "left": getattr(bbox_obj, "left", None),
                    "top": getattr(bbox_obj, "top", None),
                    "right": getattr(bbox_obj, "right", None),
                    "bottom": getattr(bbox_obj, "bottom", None),
                },
            ]

            for bbox_method in bbox_methods:
                bbox_coords = bbox_method()
                # Check if we have all required coordinates
                if all(coord is not None for coord in bbox_coords.values()):
                    # Calculate width and height
                    width = (bbox_coords["right"] or 0.0) - (bbox_coords["left"] or 0.0)
                    height = (bbox_coords["bottom"] or 0.0) - (
                        bbox_coords["top"] or 0.0
                    )

                    # Return complete bbox data
                    return {
                        "left": float(bbox_coords["left"] or 0.0),
                        "top": float(bbox_coords["top"] or 0.0),
                        "right": float(bbox_coords["right"] or 0.0),
                        "bottom": float(bbox_coords["bottom"] or 0.0),
                        "width": width,
                        "height": height,
                        "area": width * height,
                    }

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Failed to extract bbox from object: %s", e)

        return None

    def _run_docling_with_timeout(
        self,
        file_path: str,
        original_filename: str,
        timeout_seconds: int,
        job_id: Optional[str] = None,  # Pass job_id for progress updates
    ) -> List[ChunkResult]:
        """Run Docling processing with timeout using multiprocessing.

        Args:
            file_path: Path to the document file
            original_filename: Original filename without job ID prefix
            timeout_seconds: Timeout in seconds
            job_id: Job ID for progress updates (optional)

        Returns:
            List of processed chunks with metadata

        Raises:
            TimeoutError: If processing times out
            RuntimeError: If worker returns invalid result or error
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
            # Monitor progress during timeout period
            start_time = time.time()
            last_progress_update = 0
            result = None

            while result is None:
                if parent_conn.poll(1):  # Wait up to 1 second for a message
                    try:
                        message = parent_conn.recv()
                        if isinstance(message, dict):
                            if "ok" in message:
                                result = message  # Found the result
                            elif "progress" in message:
                                progress_data = message["progress"]
                                if job_id:
                                    from app.main import update_job_progress

                                    if progress_data.get("stage") == "chunking":
                                        update_job_progress(
                                            job_id,
                                            f"Processing chunks: {progress_data.get('chunks_processed', 0)}/{progress_data.get('total_chunks', 0)}",
                                            progress_data.get("progress", 20),
                                            {
                                                "current_operation": "Chunk generation",
                                                "chunks_processed": progress_data.get(
                                                    "chunks_processed", 0
                                                ),
                                                "total_chunks": progress_data.get(
                                                    "total_chunks", 0
                                                ),
                                                "stage": progress_data.get("stage"),
                                            },
                                        )
                                    elif (
                                        progress_data.get("stage")
                                        == "processing_complete"
                                    ):
                                        update_job_progress(
                                            job_id,
                                            "Document processing complete, generating embeddings",
                                            80,
                                            progress_data.get("final_stats", {}),
                                        )
                            elif "status" in message:
                                logger.info(f"Worker status: {message}")
                            else:
                                logger.debug(
                                    f"Unexpected message from worker: {message}"
                                )
                    except Exception as e:
                        logger.debug(f"Failed to process message: {e}")
                        break
                else:
                    # No message received, check for timeout
                    elapsed = time.time() - start_time
                    if elapsed >= timeout_seconds:
                        # Timeout: terminate and raise with detailed error
                        proc.terminate()
                        proc.join(timeout=5)

                        # Update progress to indicate timeout
                        if job_id:
                            from app.main import update_job_progress

                            update_job_progress(
                                job_id,
                                "Processing timed out",
                                0,
                                {
                                    "current_operation": "Timeout occurred",
                                    "timeout_seconds": timeout_seconds,
                                    "elapsed_seconds": elapsed,
                                    "error_type": "TimeoutError",
                                },
                            )

                        raise TimeoutError(
                            f"Docling processing timed out after {timeout_seconds}s for file processing"
                        )

                    # Fallback: Update progress every 30 seconds if job_id provided and no recent worker updates
                    if job_id and elapsed - last_progress_update >= 30:
                        progress = min(85, 15 + int((elapsed / timeout_seconds) * 70))
                        from app.main import update_job_progress

                        # Add more context about the timeout status
                        time_remaining = max(0, timeout_seconds - int(elapsed))
                        if time_remaining > 60:
                            status_msg = f"Processing document ({int(elapsed)} / {timeout_seconds}s, {time_remaining // 60}m remaining)"
                        else:
                            status_msg = f"Processing document ({int(elapsed)}/{timeout_seconds}s, {time_remaining}s remaining)"

                        update_job_progress(
                            job_id,
                            status_msg,
                            progress,
                            {
                                "current_operation": "Docling OCR processing",
                                "elapsed_seconds": int(elapsed),
                                "timeout_seconds": timeout_seconds,
                                "remaining_seconds": time_remaining,
                                "timeout_warning": elapsed
                                > (timeout_seconds * 0.8),  # Warn in last 20% of time
                            },
                        )
                        last_progress_update = int(elapsed)
                        logger.info(
                            f"Docling processing progress: {int(elapsed)}/{timeout_seconds}s ({progress}%)"
                        )
            logger.info(f"Docling worker result: {result}")
        finally:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
            with contextlib.suppress(Exception):
                parent_conn.close()

        if not isinstance(result, dict):
            error_msg = f"Invalid result type from Docling worker: {type(result)}"
            logger.error(error_msg)
            if job_id:
                from app.main import update_job_progress

                update_job_progress(
                    job_id,
                    "Worker communication error",
                    0,
                    {
                        "current_operation": "Worker error",
                        "error_type": "InvalidResultType",
                        "error_message": error_msg,
                    },
                )
            raise RuntimeError(error_msg)

        if not result.get("ok"):
            error_details = result.get("error", "Unknown Docling worker error")
            error_type = result.get("error_type", "UnknownError")
            traceback_info = result.get("traceback", "No traceback available")

            logger.error(
                f"Docling worker reported error [{error_type}]: {error_details}"
            )
            logger.error(f"Worker traceback: {traceback_info}")
            logger.error(f"Full worker result: {result}")

            # Provide more specific error messages based on error type
            if error_type == "ImportError":
                specific_message = f"Docling library not available: {error_details}"
            elif "CUDA" in error_details or "GPU" in error_details:
                specific_message = f"GPU/CUDA error in Docling: {error_details}"
            elif "memory" in error_details.lower():
                specific_message = (
                    f"Memory error in Docling processing: {error_details}"
                )
            elif "file" in error_details.lower():
                specific_message = f"File processing error in Docling: {error_details}"
            elif error_type == "FileNotFoundError":
                specific_message = f"Input file not found: {error_details}"
            else:
                specific_message = f"Docling processing failed: {error_details}"

            if job_id:
                from app.main import update_job_progress

                update_job_progress(
                    job_id,
                    f"Docling processing failed ({error_type})",
                    0,
                    {
                        "current_operation": "Worker error",
                        "error_type": error_type,
                        "error_message": specific_message,
                        "original_error": error_details,
                        "traceback": traceback_info,
                        "worker_result": result,
                    },
                )
            raise RuntimeError(specific_message)

        # Convert worker chunk dicts to ChunkResult with metadata
        chunks_dicts: List[Dict[str, Any]] = result.get("chunks", [])
        chunks: List[ChunkResult] = []
        for _idx, ch in enumerate(chunks_dicts, start=1):
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            # Prefer pages from worker output; otherwise try to re-collect
            pages = (
                ch.get("pages")
                if isinstance(ch, dict)
                else self._collect_chunk_pages(ch)
            )
            # Extract bounding box data (not available from worker; remains None)
            bbox_data = (
                self._collect_bounding_box(ch) if not isinstance(ch, dict) else None
            )
            chunk_id = f"chunk_{uuid.uuid4().hex}"
            meta = ChunkMetadata(
                page_num_int=pages or [1],
                original_filename=original_filename,
                chunk_size=len(text),
                chunk_overlap=0,
            )
            # Add bounding box to metadata if available
            if bbox_data:
                meta.bbox = bbox_data

            chunks.append(ChunkResult(id=chunk_id, metadata=meta, text=text))

        return chunks

    def _attach_embeddings(
        self, chunks: List[ChunkResult], job_id: Optional[str] = None
    ) -> None:
        """Generate embeddings for chunks one at a time to handle large documents.

        Args:
            chunks: List of chunks to generate embeddings for
            job_id: Job ID for progress tracking (optional)
        """
        if not chunks:
            return

        logger.info("Generating embeddings for %d chunks...", len(chunks))

        successful_embeddings = 0
        failed_embeddings = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 5  # Circuit breaker threshold

        # Track overall progress
        start_time = time.time()
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Calculate and log overall progress
            progress_percent = int((i / total_chunks) * 100)
            elapsed_time = time.time() - start_time
            avg_time_per_chunk = elapsed_time / (i + 1) if i > 0 else 0
            estimated_remaining = avg_time_per_chunk * (total_chunks - i - 1)

            logger.info(
                f"Generating embedding for chunk {i + 1}/{total_chunks} ({progress_percent}%) - "
                f"Elapsed: {elapsed_time:.1f}s, Remaining: {estimated_remaining:.1f}s"
            )

            # Log chunk information for debugging
            chunk_text_preview = (
                chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            )
            logger.info(f"Chunk text preview: {chunk_text_preview}")

            # Process one chunk at a time with detailed error handling and retries
            max_retries = 3
            base_delay = 1.0  # Base delay in seconds

            logger.info(f"length of chunk text: {len(chunk.text)}")
            for attempt in range(max_retries + 1):
                try:
                    logger.info(
                        f"Embedding attempt {attempt + 1}/{max_retries + 1} for chunk {i + 1}"
                    )

                    embedding = embedding_client.generate_embedding(chunk.text)

                    if embedding and len(embedding) > 0:
                        chunk.embeddings = embedding
                        successful_embeddings += 1
                        consecutive_timeouts = 0  # Reset circuit breaker
                        logger.info(
                            f"Successfully generated embedding for chunk {i + 1}"
                        )
                        break  # Success, exit retry loop
                    else:
                        logger.warning(
                            f"Empty or invalid embedding response for chunk {i + 1}"
                        )
                        failed_embeddings += 1
                        break  # Don't retry for empty responses

                except Exception as embed_e:
                    logger.error(
                        f"Embedding request failed for chunk {i + 1} (attempt {attempt + 1}): {type(embed_e).__name__}: {embed_e}"
                    )

                    # Check for specific HTTP errors
                    if hasattr(embed_e, "response") and embed_e.response:
                        status_code = getattr(embed_e.response, "status_code", None)
                        logger.error(
                            f"HTTP error details - Status: {status_code}, URL: {getattr(embed_e.response, 'url', 'unknown')}"
                        )

                        if status_code == 504:
                            consecutive_timeouts += 1
                            logger.warning(
                                f"Gateway timeout detected - server may be overloaded (consecutive: {consecutive_timeouts})"
                            )

                            # Circuit breaker: if too many consecutive timeouts, pause longer
                            if consecutive_timeouts >= max_consecutive_timeouts:
                                logger.error(
                                    f"Circuit breaker triggered: {consecutive_timeouts} consecutive timeouts"
                                )
                                long_pause = 30.0  # 30 second pause
                                logger.info(
                                    f"Circuit breaker: pausing for {long_pause} seconds to let service recover..."
                                )
                                time.sleep(long_pause)
                                consecutive_timeouts = 0  # Reset after long pause

                            if attempt < max_retries:
                                delay = base_delay * (2**attempt)
                                if consecutive_timeouts >= 3:
                                    delay *= 2  # Double delay for consecutive timeouts
                                logger.info(
                                    f"Retrying chunk {i + 1} in {delay} seconds..."
                                )
                                time.sleep(delay)
                                continue

                        elif status_code == 429:
                            logger.warning(
                                "Rate limit exceeded - adding delay before retry"
                            )
                            consecutive_timeouts = max(
                                0, consecutive_timeouts - 1
                            )  # Reduce consecutive timeout count
                            if attempt < max_retries:
                                delay = (
                                    base_delay * (2**attempt) * 2
                                )  # Longer delay for rate limits
                                logger.info(
                                    f"Retrying chunk {i + 1} in {delay} seconds..."
                                )
                                time.sleep(delay)
                                continue
                        elif status_code is not None and status_code >= 500:
                            logger.warning(
                                f"Server error {status_code} - embedding service may be down"
                            )
                            if attempt < max_retries and status_code in [502, 503, 504]:
                                consecutive_timeouts += 1
                                delay = base_delay * (2**attempt)
                                logger.info(
                                    f"Retrying chunk {i + 1} in {delay} seconds..."
                                )
                                time.sleep(delay)
                                continue

                    # If we've exhausted retries or it's not a retryable error
                    if attempt == max_retries:
                        logger.error(f"All retry attempts failed for chunk {i + 1}")
                        failed_embeddings += 1
                    break

            # Log progress every 10 chunks
            if (i + 1) % 10 == 0:
                progress_msg = "Processed %d/%d chunks for embeddings" % (
                    i + 1,
                    len(chunks),
                )
                logger.info(progress_msg)

                # Update progress in Redis if job_id provided
                if job_id:
                    from app.main import update_job_progress

                    progress = 80 + int(((i + 1) / len(chunks)) * 20)
                    update_job_progress(
                        job_id,
                        progress_msg,
                        progress,
                        {
                            "current_operation": "Embedding generation",
                            "chunks_processed": i + 1,
                            "total_chunks": len(chunks),
                            "successful_embeddings": successful_embeddings,
                            "failed_embeddings": failed_embeddings,
                        },
                    )

        completion_msg = "Embedding generation completed: %d successful, %d failed" % (
            successful_embeddings,
            failed_embeddings,
        )
        logger.info(completion_msg)

        if failed_embeddings > 0:
            logger.warning("%d chunks failed to generate embeddings", failed_embeddings)

        # Final progress update
        if job_id:
            from app.main import log_progress_milestone, update_job_progress

            if failed_embeddings == 0:
                update_job_progress(
                    job_id,
                    "Embeddings generated successfully",
                    100,
                    {
                        "current_operation": "Embeddings completed",
                        "successful_embeddings": successful_embeddings,
                        "total_chunks": len(chunks),
                    },
                )
                log_progress_milestone(job_id, "Embedding generation completed", 100)
            else:
                update_job_progress(
                    job_id,
                    "Embeddings completed with failures",
                    95,
                    {
                        "current_operation": "Embeddings completed with errors",
                        "successful_embeddings": successful_embeddings,
                        "failed_embeddings": failed_embeddings,
                        "total_chunks": len(chunks),
                    },
                )


def _docling_worker_process(file_path: str, conn: Connection) -> None:
    """Worker process entrypoint for Docling processing.

    Args:
        file_path: Path to the document file
        conn: Pipe connection for communication with parent process
    """
    # Set environment variables to prevent MPS issues
    if sys.platform == "darwin":
        os.environ.update(
            {"PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTORCH_MPS_DISABLE": "1"}
        )

    result = None
    try:
        logger.info(f"Worker process started for file: {file_path}")

        # Add a simple test to isolate Docling issues
        logger.info("Testing Docling imports...")

        logger.info("Docling imports successful")

        # Verify file exists and get basic info
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file does not exist: {file_path}")

        file_size = os.path.getsize(file_path)
        logger.info(f"Processing file {file_path} (size: {file_size} bytes)")

        # Test connection before starting
        conn.send({"status": "worker_started", "file": file_path, "size": file_size})
        logger.info("Worker connection test successful")

        # Test basic Docling functionality before full processing
        logger.info("Testing basic Docling DocumentConverter...")
        DocumentConverter()
        logger.info("DocumentConverter created successfully")

        # Create processor instance to use the common processing function
        logger.info("Creating DocumentProcessor instance")
        processor = DocumentProcessor()

        logger.info("Starting Docling processing with progress tracking")
        chunks_data = processor._process_with_docling(
            file_path, conn, job_id=None
        )  # Worker process doesn't have job_id context

        logger.info(
            f"Worker process completed successfully with {len(chunks_data)} chunks"
        )

        # Prepare output for pipe communication
        out_chunks: List[Dict[str, Any]] = []
        for i, chunk_data in enumerate(chunks_data):
            out_chunks.append(
                {
                    "text": chunk_data["text"],
                    "pages": chunk_data["pages"],
                    "chunk_index": i,
                }
            )

        result = {"ok": True, "chunks": out_chunks, "total_chunks": len(out_chunks)}

    except Exception as e:
        error_msg = f"Docling worker process failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        logger.exception("Full exception details:")

        # Get more detailed error information
        import traceback

        tb_str = traceback.format_exc()

        result = {
            "ok": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "traceback": tb_str,
            "file_path": file_path,
            "file_size": (
                os.path.getsize(file_path) if os.path.exists(file_path) else "unknown"
            ),
        }

    finally:
        # Always try to send the result
        if result is not None:
            try:
                conn.send(result)
                logger.info("Worker sent final result")
            except Exception as send_e:
                logger.error(f"Failed to send final result to parent: {send_e}")
        with contextlib.suppress(Exception):
            conn.close()


class FallbackDocumentProcessor:
    """Fallback processor if Docling is not available or times out."""

    def __init__(self) -> None:
        """Initialize the FallbackDocumentProcessor with supported formats."""
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
        """Process document using fallback methods.

        Args:
            file_path: Path to the document file
            params: Processing parameters dictionary
            original_filename: Original filename without job ID prefix

        Returns:
            List of processed chunks with metadata
        """
        job_id = params.get("job_id")

        try:
            if original_filename is None:
                original_filename = _extract_original_filename(file_path, params)

            # Update progress at start of fallback processing
            if job_id:
                from app.main import update_job_progress

                update_job_progress(
                    job_id,
                    f"Fallback: Extracting text from {original_filename}",
                    15,
                    {
                        "current_operation": "Fallback text extraction",
                        "fallback_method": "basic_extraction",
                    },
                )

            file_ext = os.path.splitext(file_path)[1].lower()
            content = ""
            if file_ext == ".pdf":
                logger.info("Fallback: Extracting text from PDF")
                content = self._extract_text_from_pdf(file_path, job_id)
            elif file_ext == ".docx":
                logger.info("Fallback: Extracting text from DOCX")
                content = self._extract_text_from_docx(file_path)
            elif file_ext in [".txt", ".html"]:
                logger.info("Fallback: Reading text file")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif file_ext in [".jpg", ".jpeg", ".png"]:
                logger.info("Fallback: Extracting text from image")
                content = self._extract_text_from_image(file_path)
            else:
                logger.info("Fallback: Using default content")
                content = f"Content from {original_filename}"

            # Update progress after text extraction
            if job_id:
                update_job_progress(
                    job_id,
                    "Fallback: Creating document chunks",
                    75,
                    {
                        "current_operation": "Fallback chunking",
                        "extracted_content_length": len(content),
                    },
                )

            page_info = {"original_filename": original_filename, "pages": [1]}
            chunks = self._chunk_content(content, original_filename, page_info)

            # Generate embeddings if requested
            if params.get("with_embeddings", False) and chunks:
                self._attach_embeddings(chunks, job_id)

            return chunks

        except Exception as e:
            logger.exception("Fallback processing failed for %s: %s", file_path, e)
            return [
                ChunkResult(
                    id=f"chunk_{uuid.uuid4().hex}",
                    metadata=ChunkMetadata(
                        page_num_int=[1],
                        original_filename=original_filename
                        or os.path.basename(file_path),
                    ),
                    text=f"Error processing document: {str(e)}",
                )
            ]

    def _attach_embeddings(
        self, chunks: List[ChunkResult], job_id: Optional[str] = None
    ) -> None:
        """Generate embeddings for chunks using the external embedding service.

        Args:
            chunks: List of chunks to generate embeddings for
            job_id: Job ID for progress tracking (optional)
        """
        try:
            texts = [c.text for c in chunks]

            # Use the embedding client to generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = embedding_client.generate_embeddings(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding

            # Update progress after successful embedding generation
            if job_id:
                from app.main import update_job_progress

                update_job_progress(
                    job_id,
                    "Fallback: Embeddings generated successfully",
                    95,
                    {
                        "current_operation": "Embeddings completed",
                        "embeddings_count": len(embeddings),
                    },
                )

        except Exception as e:
            logger.exception("Embedding generation failed: %s", e)

            # Update progress to indicate embedding failure
            if job_id:
                from app.main import update_job_progress

                update_job_progress(
                    job_id,
                    "Fallback: Embedding generation failed",
                    90,
                    {
                        "current_operation": "Embeddings failed",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

            # Continue without embeddings rather than failing the whole job

    def _extract_text_from_pdf(
        self, file_path: str, job_id: Optional[str] = None
    ) -> str:
        """Extract text from PDF using PyPDF2 with progress tracking.

        Args:
            file_path: Path to the PDF file
            job_id: Job ID for progress tracking (optional)

        Returns:
            Extracted text content
        """
        try:
            import PyPDF2  # type: ignore

            text = ""
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)

                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text += page_text

                    # Update progress every page or every 5 pages for large documents
                    if job_id and (i % 5 == 0 or i == total_pages - 1):
                        progress = 15 + int(
                            (i + 1) / total_pages * 60
                        )  # 15-75% range for PDF extraction
                        from app.main import update_job_progress

                        update_job_progress(
                            job_id,
                            f"Fallback: Extracting PDF text (page {i + 1}/{total_pages})",
                            progress,
                            {
                                "current_operation": "PDF text extraction",
                                "pages_processed": i + 1,
                                "total_pages": total_pages,
                                "stage": "pdf_extraction",
                            },
                        )

            return text
        except ImportError:
            logger.warning("PyPDF2 not available for PDF extraction")
            return f"PDF content would be extracted here: {os.path.basename(file_path)}"

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx.

        Args:
            file_path: Path to the DOCX file

        Returns:
            Extracted text content
        """
        try:
            import docx  # type: ignore

            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            logger.warning("python-docx not available for DOCX extraction")
            return (
                f"DOCX content would be extracted here: {os.path.basename(file_path)}"
            )

    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using Tesseract OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text content
        """
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            logger.warning(
                "Tesseract OCR not available. Install pytesseract and tesseract-ocr"
            )
            return f"Image OCR content would be extracted here: {os.path.basename(file_path)}"
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            return f"OCR extraction failed for: {os.path.basename(file_path)}"

    def _chunk_content(
        self,
        content: str,
        original_filename: str,
        page_info: dict,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> List[ChunkResult]:
        """Split content into chunks with proper metadata.

        Args:
            content: Text content to chunk
            original_filename: Original filename
            page_info: Page information dictionary
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of chunks with metadata
        """
        chunks: List[ChunkResult] = []
        start: int = 0
        n: int = len(content or "")

        while start < n:
            end: int = min(start + chunk_size, n)

            if end < n:
                break_pos = content.rfind(" ", start, end)
                if break_pos != -1 and break_pos > start + chunk_size // 2:
                    end = break_pos + 1

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunk_id = f"chunk_{uuid.uuid4().hex}"
                metadata = ChunkMetadata(
                    page_num_int=page_info.get("pages", [1]),
                    original_filename=original_filename,
                    chunk_size=len(chunk_text),
                    chunk_overlap=overlap if start > 0 else 0,
                )
                chunks.append(
                    ChunkResult(id=chunk_id, metadata=metadata, text=chunk_text)
                )

            new_start: int = end - overlap
            start = end if new_start <= start else new_start
            if start <= 0:
                start = end

        return chunks


# Create processor instance
document_processor = DocumentProcessor()

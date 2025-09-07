"""Document processing module with Docling integration."""

import contextlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple

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

    def convert_to_pdf(self, input_path: str, output_path: str) -> bool:
        """Convert document to PDF using appropriate method based on file extension.

        Args:
            input_path: Path to input file
            output_path: Path to output PDF file

        Returns:
            True if conversion succeeded, False otherwise
        """
        file_ext = os.path.splitext(input_path)[1].lower()

        # Already PDF: copy if different path
        if file_ext == ".pdf":
            logger.info(f"Bypassing conversion for PDF file: {input_path}")
            if os.path.abspath(input_path) == os.path.abspath(output_path):
                return True
            _ensure_dir(output_path)
            try:
                shutil.copy2(input_path, output_path)
                logger.info(f"Copied PDF to {output_path}")
                return True
            except Exception as e:
                logger.error("Failed to copy PDF: %s", e)
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

        # Office-like docs - unoserver first
        if file_ext in OFFICE_LIKE_EXTS:
            if self._convert_with_unoserver(input_path, output_path):
                return True
            logger.warning(
                "unoserver failed for %s; falling back to LibreOffice.", file_ext
            )
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
            logger.error("Failed reading text file: %s", e)
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
            logger.error("Failed writing temp HTML: %s", e)
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

    def _convert_with_unoserver(self, input_path: str, output_path: str) -> bool:
        """Convert document to PDF using unoconvert (client for unoserver)."""
        try:
            # Check if unoconvert is available
            if not shutil.which("unoconvert"):
                logger.debug("unoconvert not available.")
                return False

            _ensure_dir(output_path)

            # Get unoserver connection details from environment
            unoserver_host = os.getenv("UNOSERVER_HOST", "localhost")
            unoserver_port = os.getenv("UNOSERVER_PORT", "2002")

            cmd = [
                "unoconvert",
                "--host",
                unoserver_host,
                "--port",
                unoserver_port,
                "--convert-to",
                "pdf",
                input_path,
                output_path,
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=360,
            )

            if proc.returncode != 0:
                logger.warning(
                    "unoconvert failed. rc=%s, stderr=%s",
                    proc.returncode,
                    proc.stderr,
                )
                return False

            return os.path.exists(output_path)

        except Exception as e:
            logger.exception("unoconvert error: %s", e)
            return False

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
                timeout=360,
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

    def process_document(self, file_path: str, params: dict) -> Tuple[List[ChunkResult], List[str]]:
        """Process a single document using Docling's DocumentConverter and HybridChunker.

        Args:
            file_path: Path to the document file
            params: Processing parameters dictionary

        Returns:
            Tuple of processed chunks and list of temporary files to clean up
        """
        original_filename = _extract_original_filename(file_path, params)
        file_ext = os.path.splitext(file_path)[1].lower()
        temp_files = []  # Track all temporary files for cleanup

        # Bypass conversion for PDF files
        if file_ext == ".pdf":
            logger.info(f"Bypassing conversion for PDF file: {file_path}")
            processing_file_path = file_path
        else:
            upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
            temp_dir = os.path.join(upload_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            temp_pdf_path = os.path.join(temp_dir, f"converted_{uuid.uuid4().hex}.pdf")

            logger.info(f"Converting non-PDF file to PDF: {file_path}")
            conversion_success = self.document_converter.convert_to_pdf(
                file_path, temp_pdf_path
            )

            if conversion_success:
                processing_file_path = temp_pdf_path
                temp_files.append(temp_pdf_path)  # Track converted PDF for cleanup
            else:
                logger.warning(
                    "PDF conversion failed for %s, attempting direct processing",
                    file_path,
                )
                processing_file_path = file_path

        # use the multiprocessing approach with timeout
        timeout_seconds = max(10, int(params.get("timeout_seconds", 600)))

        try:
            chunks, docling_temp_files = self._run_docling_with_timeout(
                file_path=processing_file_path,
                original_filename=original_filename,
                timeout_seconds=timeout_seconds,
            )
            if not chunks:
                raise RuntimeError("Docling worker returned no chunks")
            
            # Add Docling temp files to cleanup list
            temp_files.extend(docling_temp_files)

            # Generate embeddings if requested
            if params.get("with_embeddings", False) and chunks:
                self._attach_embeddings(chunks)

            return chunks, temp_files

        except Exception as e:
            logger.error(
                "Docling processing failed for %s: %s. Falling back.", file_path, e
            )
            fallback_processor = FallbackDocumentProcessor()
            chunks, fallback_temp_files = fallback_processor.process_document(
                file_path, params, original_filename
            )
            temp_files.extend(fallback_temp_files)
            return chunks, temp_files

    def _process_with_docling(self, file_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
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
            
            # Suppress pin_memory warning when no GPU is available
            import torch
            if not torch.cuda.is_available():
                # Disable CUDA backend optimizations
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.allow_tf32 = False
                
                # Disable pin_memory in data loaders
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                
                # Monkey-patch DataLoader to force pin_memory=False
                import torch.utils.data
                original_dataloader = torch.utils.data.DataLoader
                
                def patched_dataloader(*args, **kwargs):
                    kwargs['pin_memory'] = False
                    return original_dataloader(*args, **kwargs)
                    
                torch.utils.data.DataLoader = patched_dataloader
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

            # Convert non-PDF files to PDF before processing
            if not is_pdf:
                # Create temporary PDF file in the temp subfolder
                temp_filename = f"converted_{uuid.uuid4().hex}.pdf"
                temp_pdf_path = os.path.join(temp_dir, temp_filename)

                conversion_success = self.document_converter.convert_to_pdf(
                    file_path, temp_pdf_path
                )

                if conversion_success:
                    processing_file_path = temp_pdf_path
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

            # Process the document with Docling
            converter = DocumentConverter()
            result = converter.convert(source=processing_file_path)
            doc = result.document

            # Create chunker and generate chunks
            chunker = HybridChunker()
            chunk_iter = chunker.chunk(dl_doc=doc)

            chunks_data: List[Dict[str, Any]] = []

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

            return chunks_data, [temp_pdf_path] if temp_pdf_path else []

        except Exception as e:
            logger.error("Failed to process document %s: %s", file_path, e)
            raise RuntimeError(f"Document processing failed: {e}")

        finally:
            # Clean up temporary PDF file if it was created
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    logger.debug(f"Removed temporary PDF: {temp_pdf_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary PDF {temp_pdf_path}: {e}")

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
    ) -> Tuple[List[ChunkResult], List[str]]:
        """Run Docling processing with timeout using threading.

        Args:
            file_path: Path to the document file
            original_filename: Original filename without job ID prefix
            timeout_seconds: Timeout in seconds

        Returns:
            Tuple of processed chunks and list of temporary files to clean up

        Raises:
            TimeoutError: If processing times out
            RuntimeError: If worker returns invalid result or error
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        # Run processing in a thread with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._process_with_docling, file_path)
            try:
                chunks_data, temp_files = future.result(timeout=timeout_seconds)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Docling processing timed out after {timeout_seconds}s"
                )
            except Exception as e:
                raise RuntimeError(f"Docling processing failed: {e}")

        # Convert chunk data to ChunkResult objects
        chunks: List[ChunkResult] = []
        for ch in chunks_data:
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            pages = ch.get("pages", [1])
            chunk_id = f"chunk_{uuid.uuid4().hex}"
            meta = ChunkMetadata(
                page_num_int=pages,
                original_filename=original_filename,
                chunk_size=len(text),
                chunk_overlap=0,
            )
            chunks.append(ChunkResult(id=chunk_id, metadata=meta, text=text))
            
        return chunks, temp_files

    def _attach_embeddings(self, chunks: List[ChunkResult]) -> None:
        """Generate embeddings for chunks one at a time to handle large documents.

        Args:
            chunks: List of chunks to generate embeddings for
        """
        if not chunks:
            return

        logger.info("Generating embeddings for %d chunks...", len(chunks))

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
                    logger.warning("Failed to generate embedding for chunk %d", i + 1)
                    failed_embeddings += 1

                # Log progress every 10 chunks
                if (i + 1) % 10 == 0:
                    logger.info(
                        "Processed %d/%d chunks for embeddings", i + 1, len(chunks)
                    )

            except Exception as e:
                logger.error("Error generating embedding for chunk %d: %s", i + 1, e)
                failed_embeddings += 1
                # Continue with next chunk instead of failing completely

        logger.info(
            "Embedding generation completed: %d successful, %d failed",
            successful_embeddings,
            failed_embeddings,
        )

        if failed_embeddings > 0:
            logger.warning("%d chunks failed to generate embeddings", failed_embeddings)




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
    ) -> Tuple[List[ChunkResult], List[str]]:
        """Process document using fallback methods.

        Args:
            file_path: Path to the document file
            params: Processing parameters dictionary
            original_filename: Original filename without job ID prefix

        Returns:
            Tuple of processed chunks and empty temp files list (fallback doesn't create temp files)
        """
        try:
            if original_filename is None:
                original_filename = _extract_original_filename(file_path, params)

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

            return chunks, []  # Fallback doesn't create temp files

        except Exception as e:
            logger.error("Fallback processing failed for %s: %s", file_path, e)
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
            ], []

    def _attach_embeddings(self, chunks: List[ChunkResult]) -> None:
        """Generate embeddings for chunks using the external embedding service.

        Args:
            chunks: List of chunks to generate embeddings for
        """
        try:
            texts = [c.text for c in chunks]

            # Use the embedding client to generate embeddings
            embeddings = embedding_client.generate_embeddings(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding

        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            # Continue without embeddings rather than failing the whole job

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            import PyPDF2  # type: ignore

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
        start = 0
        n = len(content or "")

        while start < n:
            end = min(start + chunk_size, n)

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

            new_start = end - overlap
            start = end if new_start <= start else new_start
            if start <= 0:
                start = end

        # Convert chunk data to ChunkResult objects
        chunks: List[ChunkResult] = []
        for ch in chunks_data:
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            pages = ch.get("pages", [1])
            chunk_id = f"chunk_{uuid.uuid4().hex}"
            meta = ChunkMetadata(
                page_num_int=pages,
                original_filename=original_filename,
                chunk_size=len(text),
                chunk_overlap=0,
            )
            chunks.append(ChunkResult(id=chunk_id, metadata=meta, text=text))
            
        return chunks, temp_files


# Create processor instance
document_processor = DocumentProcessor()

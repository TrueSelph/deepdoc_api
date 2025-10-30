"""Custom TOC chunker."""

import re
from collections import Counter
from typing import Any, Iterator

from docling_core.transforms.chunker.base import BaseChunk, BaseChunker, BaseMeta
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem
from pydantic import Field


class TocMeta(BaseMeta):
    """Metadata for TOC chunks."""

    headings: list[str] = []
    hierarchy: list[str] = []
    pages: list[int] = []


class TOCChunker(BaseChunker):
    """Custom chunker that chunks documents by sections with progress tracking."""

    section_pattern: str = Field(default=r"^(\d+(?:\.\d+)*)")

    def __init__(self) -> None:
        """Initialize the TOCChunker."""
        super().__init__()

    def process_sections(self, sections_list: list) -> list:
        """
        Process a list of sections from a document and merge sections with the same number.

        Args:
            sections_list: List of section dictionaries with name, level, page_no, section_item, and hierarchy.

        Returns:
            List of processed sections with merged names and metadata.
        """

        last_match = None
        sections: list[dict[str, Any]] = []

        for item in sections_list:
            name = item["name"].strip()
            if not name:
                continue

            match = re.match(self.section_pattern, name)

            if match:
                current_match = match.group()

                # If the section number is same as previous one, combine with the last
                if last_match == current_match and sections:
                    last_section = sections.pop()
                    remaining_text = name
                    for word in last_section["name"].split():
                        if word in remaining_text:
                            remaining_text = remaining_text.replace(word, "").strip()
                    combined_name = last_section["name"] + " " + remaining_text

                    # Merge back all metadata from last_section
                    combined_section = {**last_section, "name": combined_name}

                    sections.append(combined_section)
                else:
                    sections.append(item)
                    last_match = current_match
            else:
                sections.append(item)

        return sections

    def get_sections(
        self, dl_doc: DoclingDocument, max_repeats: int = 2
    ) -> list[tuple[str, int, int, SectionHeaderItem]]:
        """Extract all section headers from the document and return as a list of (name, header_level, page_no, item) tuples."""
        hierarchy_stack: list[SectionHeaderItem] = []
        results: list[dict] = []
        last_match = None

        # Extract all section headers
        for item, _ in dl_doc.iterate_items(with_groups=True):
            if isinstance(item, SectionHeaderItem):
                page_no = item.prov[0].page_no if item.prov else None
                sec = item.text
                match = re.match(self.section_pattern, sec)

                if match:
                    numbering = match.group(1)
                    parts = numbering.split(".")
                    depth = len(parts)

                    # Check if the current section is a continuation of the last matched section
                    if results and not results[-1]["hierarchy"] and not last_match:
                        results[-1]["hierarchy"] = hierarchy_stack[:-1]
                        last_name = results[-1]["name"]

                        # add section number to last name if missing (e.g., 3.4.1)
                        current_parts = numbering.split(".")
                        current_parts.pop()
                        last_expected_rank = int(parts[-1]) - 1
                        if last_expected_rank != 0:
                            current_parts.append(str(last_expected_rank))

                        last_name = ".".join(current_parts)
                        if last_name:
                            last_name = last_name + " " + results[-1]["name"]
                            results[-1]["name"] = last_name

                    # Trim stack to correct depth
                    hierarchy_stack = hierarchy_stack[: depth - 1]
                    hierarchy_stack.append(sec)

                    # Map to actual names, not just numbers
                    results.append(
                        {
                            "name": sec,
                            "hierarchy": hierarchy_stack[:-1],
                            "page_no": page_no,
                            "level": item.level,
                            "section_item": item,
                        }
                    )

                else:
                    results.append(
                        {
                            "name": sec,
                            "hierarchy": [],
                            "page_no": page_no,
                            "level": item.level,
                            "section_item": item,
                        }
                    )

                last_match = match

        # remove repeated sections
        section_names = []
        for item in results:
            cleaned_item = re.sub(self.section_pattern, "", item["name"]).strip()
            section_names.append(cleaned_item)

        section_counts = Counter(section_names)
        for section_name, count in section_counts.items():
            if count >= max_repeats:
                for item in results:
                    cleaned_item = re.sub(
                        self.section_pattern, "", item["name"]
                    ).strip()
                    if cleaned_item == section_name:
                        results.remove(item)

        sections = self.process_sections(results)

        return sections

    def chunk(self, dl_doc: DoclingDocument, **kwargs: object) -> Iterator[BaseChunk]:
        """Chunk the document using approved sections."""

        approved_sections = kwargs.get("approved_sections", [])
        if not approved_sections:
            approved_sections = self.get_sections(dl_doc)
        if not isinstance(approved_sections, list):
            approved_sections = []

        # Chunk only approved sections
        for section in approved_sections:
            section_name = section["name"]
            header_level = section["level"]
            page_no = section["page_no"]
            section_item = section["section_item"]
            hierarchy = section["hierarchy"]

            # Get all content under this section
            section_content = []
            current_level = header_level
            section_started = False

            for item, _ in dl_doc.iterate_items(with_groups=True):
                if item == section_item:
                    section_started = True
                    continue

                if section_started:
                    # Stop at next section of same or higher level
                    if (
                        isinstance(item, SectionHeaderItem)
                        and item.level <= current_level
                    ):
                        break

                    # Serialize the item (simple text extraction for demo)
                    if hasattr(item, "text"):
                        section_content.append(item.text)
                    elif hasattr(item, "export_to_dataframe"):
                        # For tables, convert to string
                        df = item.export_to_dataframe(dl_doc)
                        section_content.append(df.to_string())

            text_content = "\n\n".join(section_content)
            text_content = text_content.strip()
            if not text_content:
                continue

            text_content = "\n".join(hierarchy + [section_name]) + "\n" + text_content

            # Create TocMeta and BaseChunk
            meta = TocMeta(
                headings=[section_name],
                hierarchy=hierarchy,
                pages=[page_no] if page_no else [],
            )
            chunk = BaseChunk(text=text_content, meta=meta)

            yield chunk

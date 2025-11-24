import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

logger = logging.getLogger(__name__)


class LayoutProcessor:
    # formatting document
    def combine_split_pdfs(self, layouts_list):
        """
        Combines multiple layout data objects (each representing a split PDF part)
        into a single layout data object with sequential page indices.

        Args:
            layouts_list (list): A list of layout data dictionaries.

        Returns:
            dict: A combined layout data dictionary with a "pdf_info" key.
        """
        combined_data = []
        page_idx = 0

        for data in layouts_list:
            # Each data is expected to be a layout dictionary with "pdf_info"
            pdf_info = data.get("pdf_info", [])
            for page in pdf_info:
                # Update page_idx to be sequential across all parts
                page["page_idx"] = page_idx
                combined_data.append(page)
                page_idx += 1

        return {"pdf_info": combined_data}

    def process_layout(self, layout_data):
        """
        Processes the layout data to extract and chunk text content,
        handling hierarchy and ignoring repetitive headers.

        Args:
            layout_data (dict): The layout data dictionary containing "pdf_info".

        Returns:
            list: A list of formatted extracted data chunks.
        """
        extracted_data = []
        ignored_list = []

        # Get the pdf_info list
        pdf_info = layout_data.get("pdf_info", [])

        # Iterate through each page
        for page in pdf_info:
            page_idx = page.get("page_idx", 0) + 1

            # Iterate through each para_block in the page
            para_blocks = page.get("para_blocks", [])
            if not para_blocks:
                para_blocks = page.get("preproc_blocks", [])
            for item in para_blocks:
                item_type = item.get("type", "")
                lines = item.get("lines", [])

                # Process title and text items
                if item_type in ["title", "text"]:
                    for line in lines:
                        for span in line.get("spans", []):
                            data = {
                                "page": str(page_idx),
                                "content": span.get("content", ""),
                                "type": item_type
                            }
                            extracted_data.append(data)

                # Process list items
                elif item_type == "list":
                    blocks = item.get("blocks", [])
                    for block in blocks:
                        for block_line in block.get("lines", []):
                            for span in block_line.get("spans", []):
                                data = {
                                    "page": str(page_idx),
                                    "content": span.get("content", ""),
                                    "type": item_type
                                }
                                extracted_data.append(data)

                # Process table items
                elif item_type == "table":
                    blocks = item.get("blocks", [])
                    for block in blocks:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                # For tables, we might want to extract the HTML content
                                if span.get("type") == "table":
                                    table_content = span.get("html", "")
                                    # Extract text from HTML or use the image path
                                    data = {
                                        "page": str(page_idx),
                                        "content": table_content,
                                        "type": item_type,
                                        "image_path": span.get("image_path", "")
                                    }
                                    extracted_data.append(data)

            # get the headers
            para_blocks = page.get("para_blocks", [])
            if not para_blocks:
                para_blocks = page.get("preproc_blocks", [])
            for item in para_blocks:
                item_type = item.get("type", "")
                lines = item.get("lines", [])

                # Process title and text items
                if item_type in ["title", "text"]:
                    for line in lines:
                        for span in line.get("spans", []):
                            ignored_list.append(span.get("content", ""))

        # count the headers
        # Count occurrences of each item
        count = Counter(ignored_list)
        max_occurrences = 3

        # Create list of items that appear more than max_occurrences times
        ignored_list = [item for item, cnt in count.items() if cnt > max_occurrences]

        formatted_extracted_data = []
        last_data_type = ""
        main_title = False
        hierarchy = []

        # format the extracted data
        for data in extracted_data:

            if data['type'] == "title" and data['content'] not in ignored_list:

                if hierarchy and last_data_type != "title":
                    if main_title and len(hierarchy) > 1:
                        hierarchy = hierarchy[:-1]
                        if len(hierarchy) > 2:
                            hierarchy = hierarchy[-1:]

                        main_title = False
                    else:
                        hierarchy.pop()
                else:
                    main_title = True
                hierarchy.append(data['content'])
            elif (data['type'] == last_data_type and formatted_extracted_data) or (last_data_type == "text" and data["type"] == "list"):
                # Note: Added parentheses for clarity in condition above, matching original logic
                # Original: elif data['type'] == last_data_type and formatted_extracted_data or last_data_type == "text" and data["type"] == "list":
                # Python operator precedence: 'and' binds tighter than 'or'.
                # So it means: (data['type'] == last_data_type and formatted_extracted_data) OR (last_data_type == "text" and data["type"] == "list")

                last_data = formatted_extracted_data.pop()
                last_data["content"] += "\n" + data["content"]

                last_page_idx = last_data["page"]
                page_idx = str(data["page"])
                if last_page_idx != page_idx:
                    # Check if already a range
                    if " - " in last_page_idx:
                        last_page_idx = last_page_idx.split(" - ")[0]
                    last_data["page"] = last_page_idx + " - " + page_idx

                formatted_extracted_data.append(last_data)
            else:
                data['hierarchy'] = hierarchy.copy()
                hierarchy_str = " >> ".join(hierarchy)
                data['content'] = hierarchy_str + "\n" + data['content']
                formatted_extracted_data.append(data)


            last_data_type = data["type"]

        return formatted_extracted_data

    def process_single_document(self, input_file, output_file="output_single.json"):
        """
        Processes a single layout.json file.
        """

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)


        formatted_data = self.process_layout(layout_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)


    def process_split_document(self, split_folder, output_file="output_combined.json"):
        """
        Processes a folder containing split PDF layout files.
        """

        layouts_list = []
        for item in split_folder:
            path = os.path.join(item)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        layouts_list.append(data)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"Warning: layout.json not found in {item}")

        if not layouts_list:
            print("No layout data found.")
            return


        combined_layout = self.combine_split_pdfs(layouts_list)

        formatted_data = self.process_layout(combined_layout)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)


    # process document
    def run_mineru(self, input_path, output_path):
        """
        Runs the mineru command with the specified input and output paths.

        Args:
            input_path (str): Path to the input file or directory.
            output_path (str): Path to the output directory.
        """

        # Construct the command
        # Force CPU mode to prevent OOM/crash in Docker
        compute_device = os.getenv("COMPUTE_DEVICE", "cpu")
        command = ["mineru", "-p", input_path, "-o", output_path, compute_device]

        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # # Read and log output line by line
            # for line in process.stdout:
            #     print(f"Mineru: {line.strip()}", flush=True)
                
            # Wait for completion
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
                
        except subprocess.CalledProcessError as e:
            print(f"Error executing Mineru command for {input_path}:")
            # print(e.stderr) # stderr is already merged into stdout
            raise

    def process_document(self, input_path, output_base_path, max_pages=100):
        """
        Process a document with Mineru, automatically splitting it if it's too large.

        Args:
            input_path (str): Path to the input PDF file.
            output_base_path (str): Base path for output directory.
            max_pages (int): Maximum number of pages per document before splitting.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            print(f"Error: Input file {input_path} does not exist")
            return


        split_dir = []
        # Check if document needs splitting (PDF only)
        if input_path.suffix.lower() == '.pdf':
            split_output_dir = ""
            final_dir = ""
            try:
                import PyPDF2
                with open(input_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)

                    if total_pages > max_pages:

                        # Create output directories
                        document_name = input_path.stem
                        split_output_dir = Path(output_base_path) / document_name / "split_output"
                        final_dir = Path(output_base_path) / document_name / "processed.json"
                        split_docs_dir = Path(output_base_path) / document_name / "split_documents"

                        # Split and process each chunk
                        for start_page in range(0, total_pages, max_pages):
                            end_page = min(start_page + max_pages, total_pages)

                            # Create split PDF
                            pdf_writer = PyPDF2.PdfWriter()
                            for page_num in range(start_page, end_page):
                                pdf_writer.add_page(pdf_reader.pages[page_num])

                            # Save split document
                            split_filename = f"{document_name}_pages_{start_page+1}-{end_page}.pdf"
                            split_file_path = split_docs_dir / split_filename
                            split_docs_dir.mkdir(parents=True, exist_ok=True)

                            with open(split_file_path, 'wb') as output_file:
                                pdf_writer.write(output_file)

                            # Process split with Mineru
                            split_output_path = split_output_dir / f"{document_name}_pages_{start_page+1}-{end_page}"
                            self.run_mineru(str(split_file_path), str(split_output_path))

                            split_name = split_output_path.name
                            split_output_layout = split_output_path / split_name / "auto" / f"{split_name}_middle.json"
                            split_output_format_layout = split_output_dir / f"{document_name}_pages_{start_page+1}-{end_page}.json"

                            self.process_single_document(split_output_layout, split_output_format_layout)
                            split_dir.append(split_output_layout)


                        self.process_split_document(split_dir, final_dir)
                        return

            except ImportError:
                print("PyPDF2 not available. Install with: pip install PyPDF2")
            except Exception as e:
                print(f"Error processing PDF: {e}")

        # If no splitting needed or not a PDF, process normally
        document_output_dir = Path(output_base_path) / input_path.stem
        self.run_mineru(str(input_path), str(document_output_dir))
        output_layout = document_output_dir / input_path.stem / "auto" / f"{input_path.stem}_middle.json"
        output_format_layout = document_output_dir / "processed.json"

        self.process_single_document(output_layout, output_format_layout)

import os
import pymupdf  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

input_filepath = os.environ.get("INPUT_FILEPATH")
output_dir = os.environ.get("OUTPUT_DIR", "output")


def process_page(page_number):
    """Process a single page and export it as a separate PDF."""
    # Open the document
    doc = pymupdf.open(input_filepath)
    page = doc.load_page(page_number)

    # Create OCG layers for this page
    cover_ocg_xref = doc.add_ocg(f"Cover Boxes Page {page_number + 1}")
    translate_ocg_xref = doc.add_ocg(
        f"Translation Text Page {page_number + 1}")

    print(f'Processing page {page_number + 1}...')
    # Extract text blocks
    blocks = page.get_textpage().extractBLOCKS()
    total_blocks = len(blocks)
    processed_blocks = 0

    for block in blocks:
        print(
            f"Page {page_number + 1}: processing block {processed_blocks}/{total_blocks}")
        processed_blocks += 1
        # The bounding box is the first element
        bbox = block[:4]

        # Draw a white rectangle with a green outline
        page.draw_rect(
            bbox,
            color=(0, 1, 0),  # Green outline
            fill=(1, 1, 1),   # White fill
            width=0.5,        # Outline width
            overlay=True,      # Draw on top of existing content
            oc=cover_ocg_xref
        )

        css_style = """
        * {
            color: black;
        }
        """
        page.insert_htmlbox(
            bbox,
            "Sample text inserted into the rectangle.",
            css=css_style,
            scale_low=0,  # Allow unlimited scaling down to fit
            rotate=0,
            oc=translate_ocg_xref,
            opacity=1,
            overlay=True
        )

    # Save this page to a temporary file
    single_page_doc = pymupdf.open()
    single_page_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"processed_page_{page_number + 1:03d}.pdf")

    single_page_doc.save(output_path)
    single_page_doc.close()
    doc.close()

    print(f"Completed page {page_number + 1}, saved to {output_path}")


def main():
    """Main function to process pages sequentially."""
    # Open the document to get total page count
    doc = pymupdf.open(input_filepath)
    total_pages = doc.page_count
    doc.close()

    print(f"Processing {total_pages} pages sequentially...")

    # Process each page one at a time
    for page_number in range(total_pages):
        process_page(page_number)


if __name__ == "__main__":
    main()

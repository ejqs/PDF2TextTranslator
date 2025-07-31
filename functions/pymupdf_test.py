
# https://stackoverflow.com/a/53360415

import os
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()

input_filepath = os.environ.get("INPUT_FILEPATH")
output_filepath = os.environ.get(
    "OUTPUT_FILEPATH", "output/output_with_text_layer.pdf")


# Now add the text as a layer using PyMuPDF
doc = pymupdf.open(input_filepath)

cover_ocg_xref = doc.add_ocg("Cover Boxes")
translate_ocg_xref = doc.add_ocg("Translation Text")

for page in doc:
    print(f'Processing page {page.number + 1}...')
    # Extract text blocks
    blocks = page.get_textpage().extractBLOCKS()
    for block in blocks:
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
            rotate=90,
            oc=translate_ocg_xref,
            opacity=1,
            overlay=True
        )

# Save the PDF with the overlay
doc.save(output_filepath)
doc.close()

print(f"PDF with overlay saved to: {output_filepath}")

import os
import re
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_chat import AI
import asyncio
from googletrans import Translator
from deep_translator import GoogleTranslator
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn
)

load_dotenv()

input_filepath = os.environ.get("INPUT_FILEPATH")
output_dir = os.environ.get("OUTPUT_DIR", "output")
use_offline_translation = os.environ.get(
    "USE_OFFLINE_TRANSLATION", "false").lower() == "true"
rotation = int(os.environ.get(
    "ROTATION_VALUE", "0"))
debug_single_page = os.environ.get(
    "DEBUG_SINGLE_PAGE", "false").lower() == "true"
# Convert to 0-based index
# words = page.get_textpage().extractWORDS()
page_selection = os.environ.get("DEBUG_PAGE_NUMBER", "")


def parse_page_numbers(page_str):
    """Parse a string like '1;5;10-15' into a list of 0-indexed page numbers."""
    pages = set()
    if not page_str:
        return []
    parts = page_str.split(';')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    # Handle descending ranges like 20-10
                    start, end = end, start
                # User provides 1-based pages, convert to 0-based indices
                pages.update(range(start - 1, end))
            except ValueError:
                print(f"Warning: Invalid range '{part}', skipping.")
        else:
            try:
                # User provides 1-based page, convert to 0-based index
                pages.add(int(part) - 1)
            except ValueError:
                print(f"Warning: Invalid page number '{part}', skipping.")
    return sorted(list(pages))


def contains_japanese(text):
    """Check if text contains Japanese characters (Hiragana, Katakana, or Kanji)."""
    # https://stackoverflow.com/questions/6787716/regular-expression-for-japanese-characters#comment31157097_10508813
    japanese_pattern = re.compile(r'[一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤]+')
    return bool(japanese_pattern.search(text))


def extract_text_from_dict(text_dict):
    """Extract text from PyMuPDF text dictionary format."""
    text_content = []

    if "blocks" in text_dict:
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            if "text" in span:
                                text_content.append(span["text"])
                        text_content.append("\n")

    return "".join(text_content)


def extract_positioned_text_from_dict(text_dict):
    """Extract text with positioning information from PyMuPDF text dictionary format."""
    positioned_words = []

    if "blocks" in text_dict:
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            if "text" in span and span["text"].strip():
                                # Extract bounding box from span
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                text = span["text"]

                                # Create word format compatible with extractWORDS:
                                # (x0, y0, x1, y1, text, block_no, line_no, word_no)
                                positioned_word = (
                                    # x0, y0, x1, y1
                                    bbox[0], bbox[1], bbox[2], bbox[3],
                                    text,  # text content
                                    0, 0  # block_no, line_no (simplified)
                                )
                                positioned_words.append(positioned_word)

    return positioned_words


def translate_offline(text):
    """Translate Japanese text using offline translation method."""
    try:
        # First, check if the text is actually readable
        if not text or not text.strip():
            return text

        # Clean the text - remove any problematic characters
        clean_text = text.strip()
        print(f"Clean text for translation: '{clean_text[:50]}...'")

        translator = Translator()

        # Handle both sync and async versions of googletrans
        result = translator.translate(clean_text, src='ja', dest='en')

        # Check if result is a coroutine (async version)
        if asyncio.iscoroutine(result):
            # For async version, we need to run in an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(translator.translate(clean_text, src='ja', dest='en')))
                        result = future.result()
                else:
                    result = loop.run_until_complete(result)
            except RuntimeError:
                # If no event loop, create one
                result = asyncio.run(result)

        translated = result.text if hasattr(result, 'text') else str(result)
        print(f"Google Translate result: '{translated[:50]}...'")
        return translated

    except ImportError:
        print("Warning: googletrans not installed. Install with 'pip install googletrans==4.0.0rc1'")
        # Try alternative: deep-translator
        try:

            translator = GoogleTranslator(source='ja', target='en')
            translated = translator.translate(clean_text)
            print(f"Deep translator result: '{translated[:50]}...'")
            return translated
        except ImportError:
            print(
                "Alternative: Install deep-translator with 'pip install deep-translator'")
            return text
    except Exception as e:
        print(f"Offline translation error: {e}")
        print(f"Error type: {type(e)}")
        # Try alternative method
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='ja', target='en')
            translated = translator.translate(clean_text)
            print(f"Fallback deep translator result: '{translated[:50]}...'")
            return translated
        except:
            print("All translation methods failed, returning original text")
            return text


def translate_japanese_text(text, ai_instance):
    """Translate Japanese text to English using either AI chat or offline method."""
    if not contains_japanese(text.strip()):
        return text  # Return original text if no Japanese detected

    if use_offline_translation:
        print(f"Using offline translation for: {text[:50]}...")
        return translate_offline(text)
    else:
        try:
            print(f"Using AI translation for: {text[:50]}...")
            translated = ai_instance.chatgpt(
                instructions="Translate the following Japanese text to English. If the text is already in English or contains no Japanese, return it unchanged:",
                input_text=text
            )
            return translated
        except Exception as e:
            print(f"AI translation error: {e}")
            print("Falling back to offline translation...")
            return translate_offline(text)


def process_block(words, page, translate_ocg_xref, ai_instance, block_index, total_words, page_number, progress, task_id):
    """Process a single text block."""
    # The bounding box is the first element
    bbox = words[:4]

    # Extract text from the block (block[4] contains the text)
    original_text = words[4]

    # Update progress bar for each block
    progress.update(
        task_id, description=f"Page {page_number + 1}: Processing block {block_index + 1}/{total_words}")
    progress.advance(task_id)
    # Draw a white rectangle with a green outline
    # page.draw_rect(
    #     bbox,
    #     color=(1, 0, 0),  # Red outline
    #     width=0.5,        # Outline width
    #     overlay=True,      # Draw on top of existing content
    #     oc=translate_ocg_xref
    # )
    # Try to fix encoding issues
    # Save all extracted words into a .txt file
    output_txt_path = os.path.join(
        output_dir, f"extracted_words_page_{page_number + 1:03d}.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_txt_path, 'a', encoding='utf-8') as txt_file:
        txt_file.write(original_text + '\n')

    print(original_text)
    if original_text:
        try:
            # Try to encode/decode to fix potential encoding issues
            if isinstance(original_text, str):
                # Try to fix common encoding issues
                original_text = original_text.encode(
                    'utf-8', errors='ignore').decode('utf-8')

            # Remove null bytes and other problematic characters
            original_text = original_text.replace(
                '\x00', '').replace('\ufffd', '')

            # If we still have replacement characters, try a different approach
            if '�' in original_text:
                print(f"Warning: Found replacement characters in text, skipping block")
                return block_index

        except Exception as e:
            print(f"Error processing text encoding: {e}")
            return block_index

    # Debug: Print original text
    if original_text.strip():

        print(f"Original text: '{original_text[:100]}...'")
        contains_japanese_bool = contains_japanese(original_text)
        print(f"Contains Japanese: {contains_japanese_bool}")

        if contains_japanese_bool:
            page.draw_rect(
                bbox,
                color=(0, 1, 0),  # Green outline
                fill=(1, 1, 1),   # White fill
                width=0.5,        # Outline width
                overlay=True,      # Draw on top of existing content
                oc=translate_ocg_xref
            )
        else:
            page.draw_rect(
                bbox,
                color=(1, 0, 0),  # Red outline
                width=0.5,        # Outline width
                overlay=True,      # Draw on top of existing content
                oc=translate_ocg_xref
            )

            return block_index

        # Translate Japanese text if detected
        translated_text = translate_japanese_text(original_text, ai_instance)

        # Debug: Print translated text
        print(f"Translated text: '{translated_text[:100]}...'")

        css_style = """
        * {
            color: black;
            font-family: Arial, sans-serif;
        }
        """

        # Use translated text if translation occurred, otherwise use original
        display_text = translated_text if translated_text != original_text else original_text

        # Debug: Print what we're actually displaying
        print(f"Display text: '{display_text[:100]}...'")

        page.insert_htmlbox(
            bbox,
            display_text,
            css=css_style,
            scale_low=0,  # Allow unlimited scaling down to fit
            rotate=rotation,
            oc=translate_ocg_xref,
            opacity=1,
            overlay=True
        )

    return block_index  # Return for tracking completion


def process_page(page_number):
    """Process a single page and export it as a separate PDF."""
    # Initialize AI instance for this thread
    ai_instance = AI()

    # Open the original document to get the page
    doc = pymupdf.open(input_filepath)

    # Create a new single-page document
    single_page_doc = pymupdf.open()
    single_page_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)
    doc.close()  # Close original document, we only need the new one now

    # Load the page from the new document
    page = single_page_doc.load_page(0)

    # Create OCG layers directly in the new document for this page
    translate_ocg_xref = single_page_doc.add_ocg(
        f"Translation Text Page {page_number + 1}",
        on=True  # Make translation layer visible by default
    )

    print(f'Processing page {page_number + 1}...')

    # Step 1: Extract readable text using the working method
    try:
        full_text = page.get_text()
        print(
            f"Successfully extracted {len(full_text)} characters using simple extraction")

        # Save the full extracted text for reference
        output_txt_path = os.path.join(
            output_dir, f"extracted_text_page_{page_number + 1:03d}.txt")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(full_text)

    except Exception as e:
        print(f"Text extraction failed: {e}")
        full_text = ""

    # Step 2: Get positioning information using extractWORDS (even if corrupted)
    try:
        words_with_positions = page.get_textpage().extractWORDS()
        print(f"Found {len(words_with_positions)} positioned word blocks")
    except Exception as e:
        print(f"Word extraction failed: {e}")
        words_with_positions = []

    # Step 3: Try to map readable text to positions using dictionary method
    readable_words = []

    if full_text.strip() and words_with_positions:
        try:
            # Get text with detailed positioning using dictionary method
            text_dict = page.get_text("dict")
            readable_words = extract_positioned_text_from_dict(text_dict)
            print(
                f"Successfully mapped {len(readable_words)} readable text blocks with positions")

        except Exception as e:
            print(f"Dictionary extraction failed: {e}")

    # Step 4: Fallback strategy - use the positioned words even if some are corrupted
    if not readable_words and words_with_positions:
        print("Using fallback: processing positioned words even if some text is corrupted")
        # Filter out completely corrupted words (all replacement characters)
        words = []
        for word_data in words_with_positions:
            if len(word_data) > 4:
                text = word_data[4]
                # Only skip if the text is completely corrupted
                if text and not (len(text.replace('�', '').strip()) == 0):
                    words.append(word_data)
                else:
                    # Skip completely corrupted words but keep track
                    continue
        print(f"Filtered to {len(words)} words with some readable content")
    else:
        words = readable_words if readable_words else words_with_positions

    total_words = len(words)

    # Debug: Print first few words to see what we're getting
    if debug_single_page or page_number == 0:  # Enhanced debugging for single page mode or first page
        print(f"Debug: Found {total_words} words on page {page_number + 1}")
        # Show first 5 words in debug mode
        for i, block in enumerate(words[:5]):
            print(f"Block {i}: {block}")
            # Show the raw bytes to diagnose encoding issues
            if len(block) > 4:
                text = block[4]
                print(f"  Text type: {type(text)}")
                print(f"  Text repr: {repr(text[:50])}")
                if text:
                    try:
                        print(f"  Text bytes: {text.encode('utf-8')[:50]}")
                    except:
                        print("  Cannot encode text to bytes")

    # Process words concurrently within this page
    if total_words > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            page_task = progress.add_task(
                f"Processing Page {page_number + 1}", total=total_words)

            with ThreadPoolExecutor() as block_executor:
                block_futures = [
                    block_executor.submit(
                        process_block,
                        block,
                        page,
                        translate_ocg_xref,
                        ai_instance,
                        idx,
                        total_words,
                        page_number,
                        progress,
                        page_task
                    )
                    for idx, block in enumerate(words)
                ]

                for future in as_completed(block_futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(
                            f"Error processing block on page {page_number + 1}: {e}")

    # Save the modified single-page document
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"processed_page_{page_number + 1:03d}.pdf")

    single_page_doc.save(output_path, garbage=4, deflate=True, clean=True)
    single_page_doc.close()

    print(f"Completed page {page_number + 1}, saved to {output_path}")
    return page_number


def main():
    """Main function to process pages with multi-threading."""
    # Open the document to get total page count
    doc = pymupdf.open(input_filepath)
    total_pages = doc.page_count
    doc.close()

    translation_method = "offline" if use_offline_translation else "AI"
    pages_to_process = parse_page_numbers(page_selection)

    if not pages_to_process:
        # If no pages are specified in debug mode, or if not in debug mode, process all pages.
        pages_to_process = range(total_pages)

    # Filter out pages that are out of bounds
    valid_pages = [p for p in pages_to_process if p < total_pages]
    if len(valid_pages) != len(pages_to_process):
        print(
            f"Warning: Some specified pages are out of the valid range (1-{total_pages}).")
        if not valid_pages:
            print("Error: No valid pages to process.")
            return

    if debug_single_page:
        print(
            f"DEBUG MODE: Processing pages: {[p + 1 for p in valid_pages]}")
        print(f"Translation method: {translation_method}")
        # Set total for progress bar
        total_pages = len(valid_pages)
    else:
        # Only use all pages if no specific pages were requested
        if not page_selection:
            valid_pages = range(total_pages)
        print(
            f"Processing {len(valid_pages)} pages: {[p + 1 for p in valid_pages] if page_selection else 'all pages'}...")
        print(f"Translation method: {translation_method}")
        # Update total_pages for progress bar
        total_pages = len(valid_pages)

    with ThreadPoolExecutor() as executor:
        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]Processing Pages", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task(
                "Total Progress", total=total_pages)

            futures = [executor.submit(process_page, i)
                       for i in valid_pages]

            for future in as_completed(futures):
                try:
                    page_num = future.result()
                    progress.update(
                        main_task, advance=1, description=f"✓ Completed page {page_num + 1}")
                except Exception as e:
                    print(f"Error processing page: {e}")
                    progress.update(main_task, advance=1)

    print("All pages processed successfully!")


if __name__ == "__main__":
    main()

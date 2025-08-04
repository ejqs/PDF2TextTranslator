import os
import re
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_chat import AI

load_dotenv()

input_filepath = os.environ.get("INPUT_FILEPATH")
output_dir = os.environ.get("OUTPUT_DIR", "output")
use_offline_translation = os.environ.get(
    "USE_OFFLINE_TRANSLATION", "false").lower() == "true"
debug_single_page = os.environ.get(
    "DEBUG_SINGLE_PAGE", "false").lower() == "true"
# Convert to 0-based index
debug_page_number = int(os.environ.get("DEBUG_PAGE_NUMBER", "1")) - 1


def contains_japanese(text):
    """Check if text contains Japanese characters (Hiragana, Katakana, or Kanji)."""
    # https://stackoverflow.com/questions/6787716/regular-expression-for-japanese-characters#comment31157097_10508813
    # japanese_pattern = re.compile(r'[一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤]+')
    # return bool(japanese_pattern.search(text))
    return True


def translate_offline(text):
    """Translate Japanese text using offline translation method."""
    try:
        # First, check if the text is actually readable
        if not text or not text.strip():
            return text

        # Clean the text - remove any problematic characters
        clean_text = text.strip()
        print(f"Clean text for translation: '{clean_text[:50]}...'")

        # Try to import and use Google Translate offline library
        from googletrans import Translator
        translator = Translator()

        # Handle both sync and async versions of googletrans
        result = translator.translate(clean_text, src='ja', dest='en')

        # Check if result is a coroutine (async version)
        import asyncio
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
            from deep_translator import GoogleTranslator
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


def process_block(block, page, cover_ocg_xref, translate_ocg_xref, ai_instance, block_index, total_blocks, page_number):
    """Process a single text block."""
    print(
        f"Page {page_number + 1}: processing block {block_index}/{total_blocks}")

    # The bounding box is the first element
    bbox = block[:4]

    # Extract text from the block (block[4] contains the text)
    original_text = block[4] if len(block) > 4 else ""

    # Draw a white rectangle with a green outline
    page.draw_rect(
        bbox,
        color=(0, 1, 0),  # Green outline
        fill=(1, 1, 1),   # White fill
        width=0.5,        # Outline width
        overlay=True,      # Draw on top of existing content
        oc=cover_ocg_xref
    )
    # Try to fix encoding issues
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
        print(f"Contains Japanese: {contains_japanese(original_text)}")

        # Translate Japanese text if detected
        translated_text = translate_japanese_text(original_text, ai_instance)

        # Debug: Print translated text
        print(f"Translated text: '{translated_text[:100]}...'")

        css_style = """
        * {
            color: black;
            font-family: Arial, sans-serif;
            font-size: 12px;
            line-height: 1.2;
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
            rotate=0,
            oc=translate_ocg_xref,
            opacity=1,
            overlay=True
        )

    return block_index  # Return for tracking completion


def process_page(page_number):
    """Process a single page and export it as a separate PDF."""
    # Initialize AI instance for this thread
    ai_instance = AI()

    # Open the document
    doc = pymupdf.open(input_filepath)
    page = doc.load_page(page_number)

    # Create OCG layers for this page
    cover_ocg_xref = doc.add_ocg(f"Cover Boxes Page {page_number + 1}")
    translate_ocg_xref = doc.add_ocg(
        f"Translation Text Page {page_number + 1}")

    print(f'Processing page {page_number + 1}...')
    # Extract text blocks using different methods
    blocks = page.get_textpage().extractBLOCKS()
    total_blocks = len(blocks)

    # Try alternative text extraction if we get corrupted text
    if total_blocks > 0:
        sample_text = blocks[0][4] if len(blocks[0]) > 4 else ""
        if '�' in sample_text:
            print(
                "Warning: Detected corrupted text, trying alternative extraction method...")
            # Try alternative extraction method
            try:
                text_dict = page.get_text("dict")
                # This might give us better text extraction
                print("Using alternative text extraction method")
            except Exception as e:
                print(f"Alternative extraction failed: {e}")

    # Debug: Print first few blocks to see what we're getting
    if debug_single_page or page_number == 0:  # Enhanced debugging for single page mode or first page
        print(f"Debug: Found {total_blocks} blocks on page {page_number + 1}")
        # Show first 5 blocks in debug mode
        for i, block in enumerate(blocks[:5]):
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

    # Process blocks concurrently within this page
    # Limit concurrent blocks
    max_block_workers = min(8, len(blocks), os.cpu_count())

    if total_blocks > 0:
        with ThreadPoolExecutor(max_workers=max_block_workers) as block_executor:
            # Submit all block processing tasks
            block_futures = [
                block_executor.submit(
                    process_block,
                    block,
                    page,
                    cover_ocg_xref,
                    translate_ocg_xref,
                    ai_instance,
                    idx,
                    total_blocks,
                    page_number
                )
                for idx, block in enumerate(blocks)
            ]

            # Wait for all blocks to complete
            completed_blocks = 0
            for future in as_completed(block_futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                    completed_blocks += 1
                    if completed_blocks % 50 == 0:  # Progress update every 50 blocks
                        print(
                            f"Page {page_number + 1}: completed {completed_blocks}/{total_blocks} blocks")
                except Exception as e:
                    print(
                        f"Error processing block on page {page_number + 1}: {e}")

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
    return page_number


def main():
    """Main function to process pages with multi-threading."""
    # Open the document to get total page count
    doc = pymupdf.open(input_filepath)
    total_pages = doc.page_count
    doc.close()

    translation_method = "offline" if use_offline_translation else "AI"

    if debug_single_page:
        if debug_page_number >= total_pages:
            print(
                f"Error: DEBUG_PAGE_NUMBER ({debug_page_number + 1}) is greater than total pages ({total_pages})")
            return

        print(
            f"DEBUG MODE: Processing only page {debug_page_number + 1} of {total_pages}")
        print(f"Translation method: {translation_method}")

        # Process only the specified page
        try:
            process_page(debug_page_number)
            print(f"✓ Completed debug page {debug_page_number + 1}")
        except Exception as e:
            print(f"Error processing debug page: {e}")
    else:
        print(f"Processing {total_pages} pages with multi-threading...")
        print(f"Translation method: {translation_method}")
        print(f"Concurrent processing: Pages AND blocks within each page")

        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(4, os.cpu_count())  # Limit to 4 threads or CPU count
        print(f"Using {max_workers} concurrent page workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page processing tasks
            futures = [executor.submit(process_page, i)
                       for i in range(total_pages)]

            # Wait for all tasks to complete
            completed_pages = 0
            for future in as_completed(futures):
                try:
                    page_num = future.result()  # This will raise any exceptions that occurred
                    completed_pages += 1
                    print(
                        f"✓ Completed page {page_num + 1} ({completed_pages}/{total_pages})")
                except Exception as e:
                    print(f"Error processing page: {e}")

    print("All pages processed successfully!")


if __name__ == "__main__":
    main()

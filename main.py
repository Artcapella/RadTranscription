import re
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from PIL import ImageGrab
import pyperclip
import time

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- OCR FUNCTION ---
def ocr_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert for light-on-dark text
    inv = cv2.bitwise_not(gray)

    # Resize (upscale)
    height, width = inv.shape
    upscale = cv2.resize(inv, (int(width * 2), int(height * 2)), interpolation=cv2.INTER_CUBIC)

    # Threshold
    _, thresh = cv2.threshold(upscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # OCR data with bounding boxes
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(cleaned, config=custom_config, output_type=Output.DICT)
    return data

# --- SORT AND EXTRACT TEXT ---
def extract_text_ordered(img):
    data = ocr_image(img)

    n_boxes = len(data['text'])
    items = []

    for i in range(n_boxes):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            x = data['left'][i]
            y = data['top'][i]
            text = data['text'][i].strip()
            items.append((y, x, text))

    # Sort top to bottom, then left to right within line
    items.sort(key=lambda item: (item[0], item[1]))

    # Group lines by y proximity (tolerance)
    lines = []
    current_line = []
    last_y = None
    y_threshold = 10  # pixels tolerance for grouping into lines

    for y, x, text in items:
        if last_y is None or abs(y - last_y) <= y_threshold:
            current_line.append((x, text))
            last_y = y
        else:
            # Sort current line left to right
            current_line.sort(key=lambda item: item[0])
            lines.append(" ".join([t for _, t in current_line]))
            current_line = [(x, text)]
            last_y = y

    if current_line:
        current_line.sort(key=lambda item: item[0])
        lines.append(" ".join([t for _, t in current_line]))

    return lines

# --- FIX COMMON OCR ERRORS ---
# 20w5d getting read as 20wSd
def fix_common_errors(text):
    def replacement(match):
        original = match.group(0)
        corrected = f"{match.group(1)}5{match.group(2)}"
        print(f"Corrected OCR error: '{original}' → '{corrected}'")
        return corrected

    pattern = re.compile(r'(\d+w)S(d)')
    fixed_text = pattern.sub(replacement, text)
    return fixed_text

# --- TEST FUNCTION WITH DEBUG IMAGE SAVE ---
def test_image(file_path='image.png'):
    img = cv2.imread(file_path)

    # Run the same preprocessing as ocr_image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    height, width = inv.shape
    upscale = cv2.resize(inv, (int(width * 2), int(height * 2)), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(upscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Save the processed image for debugging
    debug_path = 'debug_processed.png'
    cv2.imwrite(debug_path, cleaned)
    print(f"Saved processed image to {debug_path}")

    # Continue with OCR and text extraction
    lines = extract_text_ordered(img)

    print("\n=== OCR EXTRACTED TEXT ===\n")
    for line in lines:
        line = fix_common_errors(line)
        print(line)

    # Optionally copy to clipboard
    full_text = "\n".join(lines)
    full_text = fix_common_errors(full_text)
    pyperclip.copy(full_text)
    print("\nCopied to clipboard.")


# --- MAIN LOOP FOR CLIPBOARD ---
def main_clipboard():
    print("Monitoring clipboard for images… press Ctrl+C to quit.")
    while True:
        img = ImageGrab.grabclipboard()
        if isinstance(img, ImageGrab.Image.Image):
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            lines = extract_text_ordered(arr)

            full_text = "\n".join(lines)
            pyperclip.copy(full_text)
            full_text = fix_common_errors(full_text)

            print("\n=== OCR EXTRACTED TEXT ===\n")
            print(full_text)
            print("\nCopied to clipboard.")

            time.sleep(2)  # Avoid reprocessing the same image
        time.sleep(0.5)

# --- ENTRY POINT ---
if __name__ == '__main__':
    test_image()  # Or run main_clipboard()

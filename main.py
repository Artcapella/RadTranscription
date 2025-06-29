import re
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from datetime import datetime, timedelta
import pyperclip
import time
from PIL import ImageGrab

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- CONFIGURE ---
TEMPLATE_MARKERS = {  # map field to its screen label anchor
    'BPD': 'BPD', 'HC': 'HC', 'AC': 'AC', 'FL': 'FL',
    'GA': 'GA', 'EDD_LMP': 'EDD(CUA)'  # or 'EDD(OPE)' depending
}
GA_RE = re.compile(r'(\d+)w(\d+)d')
PERCENTILE_RE = re.compile(r'(\d+\.?\d*)%')
G_FORCE = 2 * 7  # 2 weeks in days

# --- UTILS ---
def ocr_image(img):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Invert for light-on-dark text
    inv = cv2.bitwise_not(gray)

    # 3. Resize (upscale) to help Tesseract read small text better
    height, width = inv.shape
    upscale = cv2.resize(inv, (int(width * 2), int(height * 2)), interpolation=cv2.INTER_CUBIC)

    # 4. Apply Gaussian blur to reduce background noise
    blur = cv2.GaussianBlur(upscale, (5, 5), 0)

    # 5. Threshold for binarization (high contrast)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Optional morphology to clean isolated specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 7. OCR with a custom whitelist and layout config
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.%:+-/()wWdmg'

    # 8. Return full OCR data
    return pytesseract.image_to_data(cleaned, config=custom_config, output_type=Output.DICT)

LABEL_PATTERNS = {
    'BPD': re.compile(r'\bBPD\b', re.I),
    'HC': re.compile(r'\bHC\b', re.I),
    'AC': re.compile(r'\bAC\b', re.I),
    'FL': re.compile(r'\bFL\b', re.I),
    'GA': re.compile(r'GA\d+w\d+d', re.I),
    'EDD_LMP': re.compile(r'EDD\(CUA\)', re.I)
}

def find_fields(data):
    out = {}
    for i, txt in enumerate(data['text']):
        txt = txt.strip().replace(" ", "")
        for key, pattern in LABEL_PATTERNS.items():
            if pattern.search(txt):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                out[key] = (x, y, w, h)
    return out


def crop_and_read(img, bbox, expand=1.2):
    x, y, w, h = bbox
    img_height, img_width = img.shape[:2]
    
    # Calculate expansion
    ex = int(w * (expand - 1) / 2)
    ey = int(h * (expand - 1) / 2)
    
    # Ensure coordinates stay within image boundaries
    x1 = max(0, x - ex)
    y1 = max(0, y - ey)
    x2 = min(img_width, x + w + ex)
    y2 = min(img_height, y + h + ey)
    
    # Extract ROI
    roi = img[y1:y2, x1:x2]
    
    # Check if ROI is valid (not empty)
    if roi.size == 0:
        return ""
    
    text = pytesseract.image_to_string(roi, config='--psm 7')
    return text.strip()

def extract_measurements(img):
    data = ocr_image(img)
    print([t for t in data['text'] if t.strip()])
    markers = find_fields(data)
    res = {}
    for fld in ['BPD','HC','AC','FL']:
        if fld in markers:
            val = crop_and_read(img, markers[fld])
            num = re.findall(r'[\d\.]+', val)
            if num: res[fld]=float(num[0])
    # GA and EDD
    if 'GA' in markers:
        ga_txt = crop_and_read(img, markers['GA'])
        m = GA_RE.search(ga_txt)
        if m: res['GA'] = (int(m.group(1)), int(m.group(2)))
    if 'EDD_LMP' in markers:
        edd_txt = crop_and_read(img, markers['EDD_LMP'])
        try:
            dt = datetime.strptime(edd_txt, '%m/%d/%Y')
            res['EDD_LMP'] = dt
        except:
            pass
    # For EFW and percentile, may need adjacent ROI logic
    # … omitted for brevity
    return res

def compute_efw(values):
    # Simple formula (Hadlock): log10 EFW = [something] - implement or skip
    return None

def concordance_check(ga, edd_us, edd_lmp):
    if not edd_lmp or not edd_us or not ga: return ''
    diff = abs((edd_us - edd_lmp).days)
    if 14 <= (ga[0]*7 + ga[1]) <= 30 and diff <= 14:
        return 'concordant'
    return 'discordant'

def format_report(vals):
    s = []
    
    # Add measurements with GA if available
    ga = vals.get('GA')
    for fld in ['BPD','HC','AC','FL']:
        if fld in vals:
            cm = vals[fld]
            if ga:
                weeks, days = ga
                s.append(f"* {fld} = **{cm:.2f} cm**, corresponding to **{weeks} weeks {days} days** gestational age.")
            else:
                s.append(f"* {fld} = **{cm:.2f} cm**")
    
    # Add GA if available
    if ga:
        weeks, days = ga
        s.append(f"* Gestational age by ultrasound: **{weeks}w{days}d**")
    
    # Add EDD and concordance if available
    edd_us = vals.get('EDD_US')
    edd_lmp = vals.get('EDD_LMP')
    if edd_us:
        concord = concordance_check(ga, edd_us, edd_lmp)
        s.append(f"* Expected delivery date by ultrasound: **{edd_us.strftime('%d/%m/%Y')}**{', ' + concord if concord else ''}")
    
    # Add EDD LMP if available
    if edd_lmp:
        s.append(f"* Expected delivery date by LMP: **{edd_lmp.strftime('%d/%m/%Y')}**")
    
    return "\n".join(s) if s else "No measurements detected"

# --- MAIN LOOP ---
def main():
    prev = None
    print("Monitoring clipboard for images… press Ctrl+C to quit.")
    while True:
        img = ImageGrab.grabclipboard()
        if isinstance(img, ImageGrab.Image.Image):
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            data = extract_measurements(arr)
            # Compute EFW if possible
            # data['EDD_US'] = ...from GA and capture
            report = format_report(data)
            pyperclip.copy(report)
            print("Copied structured report:")
            print(report)
            time.sleep(2)
        time.sleep(0.5)

if __name__ == '__main__':
    main()

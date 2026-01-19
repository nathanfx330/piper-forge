import re
import unicodedata
from pathlib import Path

INPUT_DIR = Path("./txts")
SUFFIX = "_piperized"

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

PAUSE_WORD_LIST = [
    "but", "however", "so", "yet", "therefore", "meanwhile", 
    "instead", "because", "although", "consequently", "crucially",
    "furthermore", "conversely", "and" 
    # Added "and" strictly for testing; remove if too choppy
]

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "u.s.a", "fig", "vol", "no", "rev", "hon",
    "approx", "est"
}

def piperize_text(text: str) -> str:
    print(f"Original length: {len(text)}")
    
    # 1. NUCLEAR NORMALIZATION
    # This turns weird unicode characters (fancy quotes, weird spaces) into standard ASCII
    text = unicodedata.normalize('NFKC', text)
    
    # 2. FLATTEN
    # Turn the whole text into one long line with single spaces
    text = re.sub(r'\s+', ' ', text)

    # 3. PROTECT NUMBERS AND ABBREVIATIONS
    # We replace the punctuation in safe zones with a temporary token "___"
    
    # Protect Numbers (10,000 -> 10___000)
    text = re.sub(r'(\d),(\d)', r'\1__COMMA__\2', text)
    text = re.sub(r'(\d)\.(\d)', r'\1__DOT__\2', text)
    text = re.sub(r'(\d):(\d)', r'\1__COLON__\2', text)
    
    # Protect Abbreviations (Dr. -> Dr___)
    abbr_pattern = r'\b(' + '|'.join(re.escape(a) for a in ABBREVIATIONS) + r')\.'
    def mask_abbr(m): return m.group(0).replace(".", "__DOT__")
    text = re.sub(abbr_pattern, mask_abbr, text, flags=re.IGNORECASE)

    # 4. DESTROY PUNCTUATION
    # Now that safe commas are hidden, KILL ALL REMAINING COMMAS
    
    # Explicitly check if we are finding commas
    comma_count = text.count(',')
    print(f"Found {comma_count} unprotected commas to split.")

    # Replace Comma, Semicolon, Colon with Breaks
    text = text.replace(",", ", ...\n\n")
    text = text.replace(";", "; ...\n\n")
    text = text.replace(":", ": ...\n\n")
    
    # Replace Periods (Sentence ends)
    text = text.replace(".", ". ...\n\n")
    text = text.replace("?", "? ...\n\n")
    text = text.replace("!", "! ...\n\n")

    # Replace Dashes (Space - Space)
    text = text.replace(" - ", " ...\n\n")
    text = text.replace("—", " ...\n\n")

    # 5. TRANSITION WORDS
    # Add a break before specific words
    pattern = r"\s+\b(" + "|".join(PAUSE_WORD_LIST) + r")\b"
    text = re.sub(pattern, r' ...\n\n\1', text, flags=re.IGNORECASE)

    # 6. RESTORE PROTECTED CONTENT
    text = text.replace("__COMMA__", ",")
    text = text.replace("__DOT__", ".")
    text = text.replace("__COLON__", ":")

    # 7. CLEANUP
    # Fix double ellipses or excessive newlines
    text = text.replace(".......", "...")
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def main():
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {INPUT_DIR}")
        return

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found.")
        return

    print("\nAvailable text files:")
    for i, f in enumerate(txt_files, start=1):
        print(f"{i}: {f.name}")

    try:
        choice = int(input("\nSelect file number: ").strip())
        selected_file = txt_files[choice - 1]
    except Exception:
        print("Invalid selection.")
        return

    output_file = selected_file.with_name(selected_file.stem + SUFFIX + selected_file.suffix)

    with selected_file.open("r", encoding="utf-8") as f:
        original = f.read()

    piperized = piperize_text(original)

    with output_file.open("w", encoding="utf-8") as f:
        f.write(piperized)

    print(f"\nWritten to: {output_file.name}")
    print("-" * 40)
    print("PREVIEW (First 500 chars):")
    print(piperized[:500])
    print("-" * 40)

if __name__ == "__main__":
    main()

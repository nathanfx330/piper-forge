import re
import unicodedata
from pathlib import Path

INPUT_DIR = Path("./txts")
SUFFIX = "_piperized"

# Words to force a pause before (Transition words)
PAUSE_WORD_LIST = [
    "but", "however", "so", "yet", "therefore", "meanwhile", 
    "instead", "because", "although", "consequently", "crucially",
    "furthermore", "conversely", "and"
]

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "u.s.a", "fig", "vol", "no", "rev", "hon",
    "approx", "est"
}

def piperize_text(text: str) -> str:
    print(f"Original length: {len(text)}")
    
    # 1. Normalize & Flatten
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text) # Turn newlines/tabs into single spaces

    # 2. PROTECT (Hide things we don't want to break)
    
    # Numbers (10,000)
    text = re.sub(r'(\d),(\d)', r'\1__COMMA__\2', text)
    # Decimals (3.14)
    text = re.sub(r'(\d)\.(\d)', r'\1__DOT__\2', text)
    # Time (12:30)
    text = re.sub(r'(\d):(\d)', r'\1__COLON__\2', text)
    
    # Abbreviations (Dr.)
    abbr_pattern = r'\b(' + '|'.join(re.escape(a) for a in ABBREVIATIONS) + r')\.'
    def mask_abbr(m): return m.group(0).replace(".", "__DOT__")
    text = re.sub(abbr_pattern, mask_abbr, text, flags=re.IGNORECASE)

    # URLs
    def mask_url(m): return m.group(0).replace(".", "__DOT__").replace(":", "__COLON__")
    text = re.sub(r'(https?://\S+|www\.\S+)', mask_url, text)

    # 3. APPLY PAUSES (The "..." insertion)
    
    # Explicitly print what we are doing for debugging
    print("Inserting pauses...")

    # Commas -> ", ..."
    text = text.replace(",", ", ...")
    
    # Semicolons -> "; ..."
    text = text.replace(";", "; ...")
    
    # Colons -> ": ..." (User specifically asked for this)
    text = text.replace(":", ": ...")
    
    # Periods -> ". ..."
    text = text.replace(".", ". ...")
    text = text.replace("?", "? ...")
    text = text.replace("!", "! ...")

    # Dashes -> "..."
    text = text.replace(" - ", " ... ")
    text = text.replace("—", " ... ")

    # Transition Words (add ... before them)
    # We use a space before the ... to ensure it doesn't merge with previous word
    pattern = r"\s+\b(" + "|".join(PAUSE_WORD_LIST) + r")\b"
    text = re.sub(pattern, r' ... \1', text, flags=re.IGNORECASE)

    # 4. RESTORE PROTECTED CHARS
    text = text.replace("__COMMA__", ",")
    text = text.replace("__DOT__", ".")
    text = text.replace("__COLON__", ":")

    # 5. CLEANUP
    # Ensure we don't have double spaces or "....... "
    text = text.replace(".......", "...")
    text = text.replace("....", "...")
    text = text.replace("... ...", "...")
    
    # Ensure there is a space after every ellipsis
    text = text.replace("...", "... ")
    
    # Final Flatten to ensure absolutely ONE line
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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

import re
from pathlib import Path

INPUT_DIR = Path("./txts")
SUFFIX = "_piperized"

DISCOURSE_WORDS = [
    "well",
    "so",
    "now",
    "then",
    "anyway",
    "however",
    "therefore",
    "meanwhile",
]

CONJUNCTIONS = [
    "and",
    "but",
    "or",
    "so",
    "yet",
]


def piperize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)

    for word in DISCOURSE_WORDS:
        text = re.sub(
            rf"\b({word})\b(?!,)",
            r"\1,",
            text,
            flags=re.IGNORECASE,
        )

    for conj in CONJUNCTIONS:
        text = re.sub(
            rf"(?<![,—])\s+\b({conj})\b",
            r", \1",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r",(\S)", r", \1", text)

    text = re.sub(
        r"\b(I mean|you know|that is)\b",
        r"\1…",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\.(\s|$)", r"...\1", text)

    return text.strip()


def main():
    if not INPUT_DIR.exists():
        print(f"Directory not found: {INPUT_DIR}")
        return

    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print("No .txt files found.")
        return

    print("\nAvailable text files:\n")
    for i, f in enumerate(txt_files, start=1):
        print(f"{i}: {f.name}")

    try:
        choice = int(input("\nChoose a file number to piperize: ").strip())
        if choice < 1 or choice > len(txt_files):
            raise ValueError
    except ValueError:
        print("Invalid selection.")
        return

    selected_file = txt_files[choice - 1]
    output_file = selected_file.with_name(
        selected_file.stem + SUFFIX + selected_file.suffix
    )

    with selected_file.open("r", encoding="utf-8") as f:
        original = f.read()

    piperized = piperize_text(original)

    with output_file.open("w", encoding="utf-8") as f:
        f.write(piperized)

    print(f"\nPiperized file written:")
    print(f"  {output_file.name}")


if __name__ == "__main__":
    main()

import os
import pdfplumber

DATA_DIR = "data"

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(DATA_DIR, filename)
    txt_filename = filename.replace(".pdf", ".txt")
    txt_path = os.path.join(DATA_DIR, txt_filename)

    print(f"Converting: {filename}")

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(
                x_tolerance=2,
                y_tolerance=2
            )
            if page_text:
                text += page_text + "\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

print("PDF conversion to text completed.")
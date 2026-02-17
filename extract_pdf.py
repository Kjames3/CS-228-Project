import pypdf
import sys

def extract_text(pdf_path, txt_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for page in reader.pages:
                f.write(page.extract_text() + "\n")
        print(f"Successfully extracted text to {txt_path}")
    except Exception as e:
        print(f"Error extracting text: {e}")
        sys.exit(1)

if __name__ == "__main__":
    extract_text("documents/Project_Proposal_CS228.pdf", "documents/Project_Proposal_CS228.txt")

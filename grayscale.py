import fitz  # PyMuPDF
from PIL import Image
import os

def pdf_to_grayscale(pdf_path, output_folder):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Get the number of pages
    num_pages = len(pdf_document)
    print(f"Number of pages in PDF: {num_pages}")

    if num_pages == 0:
        print("The PDF has no pages.")
        return

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each page in the PDF
    for page_number in range(num_pages):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()

        # Convert pixmap to a PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to grayscale and save
        gray_img = img.convert("L")  # Convert to grayscale
        grayscale_image_path = os.path.join(output_folder, f"page_{page_number + 1}_grayscale.png")
        gray_img.save(grayscale_image_path)

    print("All PDF pages have been converted to grayscale PNG images.")

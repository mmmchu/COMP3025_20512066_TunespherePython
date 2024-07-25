import fitz  # PyMuPDF
from PIL import Image
import os

def pdf_to_grayscale_and_binarize(pdf_path, output_folder, threshold= 195):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Check if the PDF has at least one page
    if len(pdf_document) == 0:
        print("The PDF has no pages.")
        return

    # Process only the first page (page index 0)
    page_number = 0
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()

    # Convert pixmap to a PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert to grayscale
    gray_img = img.convert("L")

    # Binarize the grayscale image
    binarized_img = gray_img.point(lambda p: p > threshold and 255)

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the grayscale image
    grayscale_image_path = os.path.join(output_folder,
                                        f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{page_number + 1}_grayscale.png")
    gray_img.save(grayscale_image_path)

    # Save the binarized image
    binarized_image_path = os.path.join(output_folder,
                                        f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{page_number + 1}_binarized.png")
    binarized_img.save(binarized_image_path)

    print(f"Page 1 of {pdf_path} has been converted to grayscale and binarized images.")

# Example usage
pdf_path = 'Image/music1.pdf'
output_folder = 'processed_images'
pdf_to_grayscale_and_binarize(pdf_path, output_folder)

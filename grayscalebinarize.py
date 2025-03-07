import fitz  # PyMuPDF
from PIL import Image
import os

def pdf_to_grayscale_and_binarize(pdfpath, outputfolder, threshold=185):
    print(f"Processing PDF: {pdfpath}")

    # Open the PDF file
    pdf_document = fitz.open(pdfpath)

    # Check if the PDF has at least one page
    if len(pdf_document) == 0:
        print("The PDF has no pages.")
        return None

    # Process only the first page (page index 0)
    page_number = 0
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    # Assuming pix is an object that has width, height, and samples attributes
    width, height = pix.width, pix.height
    mode = "RGB"
    data = pix.samples

    print(f"Loaded page {page_number + 1} from the PDF.")

    # Convert pixmap to a PIL Image
    # Convert width and height to a tuple
    img = Image.frombytes(mode, (width, height), data)
    print(f"Original image size: {img.size}")

    # Convert to grayscale
    gray_img = img.convert("L")
    print("Converted image to grayscale.")

    # Binarize the grayscale image
    binarized_img = gray_img.point(lambda p: p > threshold and 255)
    print(f"Binarized image with threshold {threshold}.")

    # Create the output folder if it does not exist
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
        print(f"Created output folder: {outputfolder}")

    # Save the grayscale image
    grayscale_image_path = os.path.join(outputfolder,
                                        f"{os.path.basename(pdfpath).replace('.pdf', '')}_pg_{page_number + 1}_GS.png")
    gray_img.save(grayscale_image_path)
    print(f"Saved grayscale image to: {grayscale_image_path}")

    # Save the binarized image
    binarizedimagepath = os.path.join(outputfolder,
                                      f"{os.path.basename(pdfpath).replace('.pdf', '')}_pg_{page_number + 1}_BN.png")
    binarized_img.save(binarizedimagepath)
    print(f"Saved binarized image to: {binarizedimagepath}")

    return binarizedimagepath


# Example usage
pdf_path = 'Image/music1.pdf'
output_folder = 'processed_images'
binarized_image_path = pdf_to_grayscale_and_binarize(pdf_path, output_folder)

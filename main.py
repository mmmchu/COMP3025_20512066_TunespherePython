from grayscale import pdf_to_grayscale

# Path to your PDF file
pdf_path = 'Image/music1.pdf'

# Folder where grayscale PNG images will be saved
output_folder = 'processed_images'

# Convert PDF pages to grayscale PNG images
pdf_to_grayscale(pdf_path, output_folder)

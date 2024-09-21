import cv2
import pytesseract
import csv
import os
import numpy as np

# Set the mode
real_time_mode = True
#real_time_mode = False

# Specify the path to the image file
image_path = "GrayCO2.JPEG"

#OCR = True
OCR = False

# Initialize the camera (only used in real-time mode)
if real_time_mode:
    cap = cv2.VideoCapture(5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define the target string
target_string = "0123456789AB"

def update(val):
    pass

# Function to save parameters to a CSV file
def save_parameters(block_size, c, blur_kernel_size, morph_kernel_size, threshold_type):
    print("Saving parameters...")  # Debugging statement
    with open('parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Block Size', 'C', 'Blur Kernel Size', 'Morph Kernel Size', 'Threshold Type'])
        writer.writerow([block_size, c, blur_kernel_size, morph_kernel_size, threshold_type])
    print(f"Saved parameters: Block Size={block_size}, C={c}, Blur Kernel Size={blur_kernel_size}, Morph Kernel Size={morph_kernel_size}, Threshold Type={threshold_type}")  # Debugging statement

# Function to load parameters from a CSV file
def load_parameters():
    if os.path.exists('parameters.csv'):
        with open('parameters.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            params = next(reader)
            print(f"Loaded parameters: {params}")  # Debugging statement
            return int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4])
    print("Loading default parameters...")  # Debugging statement
    return 11, 2, 5, 3, 0  # Default values

# Load parameters before creating trackbars
block_size, c, blur_kernel_size, morph_kernel_size, threshold_type = load_parameters()

# Create a window
cv2.namedWindow('OCR Real-time Threshold Video')

# Create trackbars for block size, constant, and GaussianBlur and morphology kernel sizes
cv2.createTrackbar('Block Size', 'OCR Real-time Threshold Video', block_size, 255, update)
cv2.createTrackbar('C', 'OCR Real-time Threshold Video', c, 100, update)
cv2.createTrackbar('Blur Kernel Size', 'OCR Real-time Threshold Video', blur_kernel_size, 50, update)
cv2.createTrackbar('Morph Kernel Size', 'OCR Real-time Threshold Video', morph_kernel_size, 50, update)
cv2.createTrackbar('Threshold Type', 'OCR Real-time Threshold Video', threshold_type, 1, update)  # 0 for THRESH_BINARY, 1 for THRESH_BINARY_INV

# Function to remove noise from the image
def remove_noise(image, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

# Function to apply morphological operations to the binary image
def apply_morphology(binary_image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morphed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return morphed

def process_image(image):
    # Get current positions of trackbars
    block_size = cv2.getTrackbarPos('Block Size', 'OCR Real-time Threshold Video')
    c = cv2.getTrackbarPos('C', 'OCR Real-time Threshold Video')
    blur_kernel_size = cv2.getTrackbarPos('Blur Kernel Size', 'OCR Real-time Threshold Video')
    morph_kernel_size = cv2.getTrackbarPos('Morph Kernel Size', 'OCR Real-time Threshold Video')
    threshold_type = cv2.getTrackbarPos('Threshold Type', 'OCR Real-time Threshold Video')

    # Ensure block_size is odd and greater than 1
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3
    if morph_kernel_size < 3:
        morph_kernel_size = 3

    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise from the grayscale image
    denoised = remove_noise(gray, blur_kernel_size)

    # Apply adaptive thresholding to preprocess the image
    threshold_method = cv2.THRESH_BINARY if threshold_type == 0 else cv2.THRESH_BINARY_INV
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_method, block_size, c)

    # Apply morphological operations to the binary image
    morphed = apply_morphology(thresh, morph_kernel_size)

    # Combine the original grayscale image and the morphed image side by side
    combined = np.hstack((gray, morphed))

    if OCR:
        # Perform OCR on the processed frame
        custom_config = r'--oem 3 --psm 6'
        ocr_result = pytesseract.image_to_string(morphed, config=custom_config)

        # Remove any extra characters or spaces
        detected_text = ''.join(filter(str.isalnum, ocr_result))

        # Get bounding box information for each character
        boxes = pytesseract.image_to_boxes(morphed, config=custom_config)

        # Overlay rectangles on the detected characters on the original grayscale image
        h, w = gray.shape
        for box in boxes.splitlines():
            box = box.split()
            x, y, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            y = h - y  # Flip y-coordinate to match OpenCV's coordinate system
            y2 = h - y2
            cv2.rectangle(combined, (x, y), (x2, y2), (255, 0, 0), 1)  # Draw on the original image with blue color

        # Compare the detected text with the target string
        match_result = "OK" if detected_text == target_string else "Not match"

        # Display the detected text and the result on the combined image
        cv2.putText(combined, f"Detected: {detected_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"Result: {match_result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the combined image
    cv2.imshow('OCR Real-time Threshold Video', combined)

def main():
    if real_time_mode:
        while True:
            for _ in range(1):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                break

            process_image(frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            while True:
                process_image(image)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # After the loop exits, ensure final parameters are saved
            block_size = cv2.getTrackbarPos('Block Size', 'OCR Real-time Threshold Video')
            c = cv2.getTrackbarPos('C', 'OCR Real-time Threshold Video')
            blur_kernel_size = cv2.getTrackbarPos('Blur Kernel Size', 'OCR Real-time Threshold Video')
            morph_kernel_size = cv2.getTrackbarPos('Morph Kernel Size', 'OCR Real-time Threshold Video')
            threshold_type = cv2.getTrackbarPos('Threshold Type', 'OCR Real-time Threshold Video')
            save_parameters(block_size, c, blur_kernel_size, morph_kernel_size, threshold_type)
        else:
            print("Image file not found.")

    if real_time_mode:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

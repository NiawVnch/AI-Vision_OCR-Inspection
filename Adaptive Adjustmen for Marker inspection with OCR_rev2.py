import cv2
import pytesseract
import csv
import os
import numpy as np
import socket

# Set the mode
real_time_mode = True
# real_time_mode = False

# Specify the path to the image file
image_path = "GrayCO2.JPEG"

OCR = True
# OCR = False

# Initialize the camera (only used in real-time mode)
if real_time_mode:
    cap = cv2.VideoCapture(5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define the target string
target_string = "0123456789AB"

# Function to update trackbars
def update(val):
    pass

# Function to save parameters to a CSV file
def save_parameters(block_size, c, blur_kernel_size, morph_kernel_size, threshold_type):
    print("Saving parameters...")  # Debugging statement
    with open('parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Crop Coordinate', 'Block Size', 'C', 'Blur Kernel Size', 'Morph Kernel Size', 'Threshold Type'])
        writer.writerow([str(crop_coords), block_size, c, blur_kernel_size, morph_kernel_size, threshold_type])
    print(f"Saved parameters: Block Size={block_size}, C={c}, Blur Kernel Size={blur_kernel_size}, Morph Kernel Size={morph_kernel_size}, Threshold Type={threshold_type}")  # Debugging statement

# Function to load parameters from a CSV file
def load_parameters():
    if os.path.exists('parameters.csv'):
        with open('parameters.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            params = next(reader)
            crop_coords = eval(params[0])  # Evaluate the string as a tuple
            print(f"Loaded parameters: {params}")  # Debugging statement
            return crop_coords, int(params[1]), int(params[2]), int(params[3]), int(params[4]), int(params[5])
    print("Loading default parameters...")  # Debugging statement
    return (100, 100, 200, 200), 11, 2, 5, 3, 0  # Default values

# Load parameters before creating trackbars
crop_coords, block_size, c, blur_kernel_size, morph_kernel_size, threshold_type = load_parameters()

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

# Function to send OCR result to server via TCP/IP
def send_ocr_result(result):
    server_ip = "192.168.0.100"  # Replace with actual server IP
    server_port = 12345  # Replace with the actual port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)  # Set a 5-second timeout for the socket connection
            s.connect((server_ip, server_port))
            s.sendall(result.encode())
            print(f"Sent OCR result to server: {result}")
    except (socket.timeout, socket.error) as e:
        print(f"Error: Could not send OCR result to server. {e}")

def process_image(image):
    # Initialize detected_text to an empty string in case OCR is not enabled
    detected_text = ""

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

    # Crop the image using loaded coordinates
    x1, y1, x2, y2 = crop_coords
    cropped = image[y1:y2, x1:x2]

    # Convert the cropped frame to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Remove noise from the grayscale image
    denoised = remove_noise(gray, blur_kernel_size)

    # Apply adaptive thresholding to preprocess the image
    threshold_method = cv2.THRESH_BINARY if threshold_type == 0 else cv2.THRESH_BINARY_INV
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_method, block_size, c)

    # Apply morphological operations to the binary image
    morphed = apply_morphology(thresh, morph_kernel_size)

    # Draw crop area on the original frame (instead of the cropped image) for real-time reference
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle for the crop region

    # Combine the original image and the thresholded image side by side
    original_with_rect = image.copy()  # Copy the original image to add the rectangle without modifying it
    resized_thresh = cv2.resize(morphed, (image.shape[1], image.shape[0]))  # Resize thresholded image to match
    threshold_bgr = cv2.cvtColor(resized_thresh, cv2.COLOR_GRAY2BGR)  # Convert threshold to BGR for side by side
    combined = np.hstack((original_with_rect, threshold_bgr))  # Combine images side by side

    if OCR:
        # Perform OCR on the processed cropped frame
        custom_config = r'--oem 3 --psm 6'
        ocr_result = pytesseract.image_to_string(morphed, config=custom_config)

        # Remove any extra characters or spaces
        detected_text = ''.join(filter(str.isalnum, ocr_result))

        # Compare the detected text with the target string
        match_result = "OK" if detected_text == target_string else "Not match"

        # Display the detected text and the result on the combined image
        cv2.putText(combined, f"Detected: {detected_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"Result: {match_result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the combined original and thresholded images side by side
    cv2.imshow('OCR Real-time Threshold Video', combined)

    return detected_text

def main():
    detected_text = ""
    try:
        if real_time_mode:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detected_text = process_image(frame)

                # Capture result when 'c' is pressed
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    print(f"OCR Capture: {detected_text}")
                    if detected_text:
                        send_ocr_result(detected_text)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                while True:
                    detected_text = process_image(image)

                    # Capture result when 'c' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        print(f"OCR Capture: {detected_text}")
                        if detected_text:
                            send_ocr_result(detected_text)

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

    finally:
        # Release resources even if an error occurs
        if real_time_mode and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

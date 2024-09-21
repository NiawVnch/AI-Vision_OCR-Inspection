import cv2
import pytesseract
import csv
import os
import numpy as np
import socket
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# TCP/IP Server address and port
server_ip = "127.0.0.1"
server_port = 12345

# Load parameters from CSV
def load_parameters():
    if os.path.exists('parameters.csv'):
        with open('parameters.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            params = next(reader)
            print(f"Loaded parameters: {params}")
            crop_coords = eval(params[0])  # Convert string to tuple
            return crop_coords, int(params[1]), int(params[2]), int(params[3]), int(params[4]), int(params[5])
    print("Loading default parameters...")
    return (0, 0, 640, 480), 11, 2, 5, 3, 0  # Default values

# Save parameters to CSV
def save_parameters(crop_coords, block_size, c, blur_kernel_size, morph_kernel_size, threshold_type):
    print("Saving parameters...")
    with open('parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Crop Coordinate', 'Block Size', 'C', 'Blur Kernel Size', 'Morph Kernel Size', 'Threshold Type'])
        writer.writerow([crop_coords, block_size, c, blur_kernel_size, morph_kernel_size, threshold_type])
    print(f"Saved parameters: {crop_coords}, Block Size={block_size}, C={c}, Blur Kernel Size={blur_kernel_size}, Morph Kernel Size={morph_kernel_size}, Threshold Type={threshold_type}")

# Initialize parameters
crop_coords, block_size, c, blur_kernel_size, morph_kernel_size, threshold_type = load_parameters()

# Function to send OCR result to the server via TCP/IP
def send_ocr_result(result):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip, server_port))
            s.sendall(result.encode())
            print(f"Sent OCR result to server: {result}")
    except ConnectionError as e:
        print(f"Error sending OCR result: {e}")

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

# Function to process and perform OCR on the image
def process_image(image):
    global crop_coords, block_size, c, blur_kernel_size, morph_kernel_size, threshold_type

    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3
    if morph_kernel_size < 3:
        morph_kernel_size = 3

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_gray = gray[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]

    denoised = remove_noise(cropped_gray, blur_kernel_size)
    threshold_method = cv2.THRESH_BINARY if threshold_type == 0 else cv2.THRESH_BINARY_INV
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_method, block_size, c)
    morphed = apply_morphology(thresh, morph_kernel_size)

    combined = np.hstack((cropped_gray, morphed))

    custom_config = r'--oem 3 --psm 6'
    ocr_result = pytesseract.image_to_string(morphed, config=custom_config)
    detected_text = ''.join(filter(str.isalnum, ocr_result))

    return detected_text, combined

# Capture frame and process OCR
def capture_frame():
    global frame
    detected_text, processed_frame = process_image(frame)
    print(f"Captured text: {detected_text}")
    send_ocr_result(detected_text)
    cv2.imshow('Captured Image', processed_frame)

# Main loop for the real-time video feed
def update_video_feed():
    global frame, cap
    ret, frame = cap.read()
    if ret:
        # Draw crop rectangle on the frame
        cv2.rectangle(frame, (crop_coords[0], crop_coords[1]), (crop_coords[2], crop_coords[3]), (0, 255, 0), 2)

        # Convert frame to Tkinter-compatible image and display in the GUI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, update_video_feed)

# Initialize OpenCV trackbars for adjusting parameters in real-time
def setup_opencv_trackbars():
    cv2.namedWindow('Trackbars')
    
    # Trackbars for adjusting the parameters
    cv2.createTrackbar('Block Size', 'Trackbars', block_size, 255, lambda x: None)
    cv2.createTrackbar('C', 'Trackbars', c, 100, lambda x: None)
    cv2.createTrackbar('Blur Kernel Size', 'Trackbars', blur_kernel_size, 50, lambda x: None)
    cv2.createTrackbar('Morph Kernel Size', 'Trackbars', morph_kernel_size, 50, lambda x: None)
    cv2.createTrackbar('Threshold Type', 'Trackbars', threshold_type, 1, lambda x: None)

# Update the parameters from OpenCV trackbars
def update_parameters():
    global block_size, c, blur_kernel_size, morph_kernel_size, threshold_type
    
    # Get current positions of trackbars
    block_size = cv2.getTrackbarPos('Block Size', 'Trackbars')
    c = cv2.getTrackbarPos('C', 'Trackbars')
    blur_kernel_size = cv2.getTrackbarPos('Blur Kernel Size', 'Trackbars')
    morph_kernel_size = cv2.getTrackbarPos('Morph Kernel Size', 'Trackbars')
    threshold_type = cv2.getTrackbarPos('Threshold Type', 'Trackbars')

    # Ensure trackbars update the parameters
    update_video_feed()
    video_label.after(100, update_parameters)  # Keep updating the parameters every 100ms

# Initialize the Tkinter GUI
def setup_gui():
    global root, video_label
    root = tk.Tk()
    root.title("OCR Real-time Video Feed")

    # Create a label in the GUI to display video
    video_label = tk.Label(root)
    video_label.pack()

    # Create Capture button
    capture_button = ttk.Button(root, text="Capture", command=capture_frame)
    capture_button.pack()

    # Start updating the video feed
    update_video_feed()

    # Update OpenCV trackbar parameters in real-time
    update_parameters()

    # Start the Tkinter event loop
    root.mainloop()

# Initialize the video capture
def main():
    global cap
    cap = cv2.VideoCapture(5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    setup_opencv_trackbars()  # Initialize OpenCV trackbars for parameter adjustment
    setup_gui()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

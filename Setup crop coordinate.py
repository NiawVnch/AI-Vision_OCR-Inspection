import cv2
import csv
import os

# File to store crop coordinates and other parameters
csv_file = 'parameters.csv'

# Default parameters (will be overwritten if the CSV exists)
crop_coordinates = (0, 0, 100, 100)  # Default crop coordinates
block_size = 54
C = 14
blur_kernel_size = 6
morph_kernel_size = 3
threshold_type = 1

# Load parameters from CSV file if it exists
if os.path.exists(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            crop_coordinates = eval(row[0])
            block_size = int(row[1])
            C = int(row[2])
            blur_kernel_size = int(row[3])
            morph_kernel_size = int(row[4])
            threshold_type = int(row[5])

# Global variables for the crop rectangle
start_x, start_y, end_x, end_y = crop_coordinates
drawing = False

# Function to update CSV file with new parameters
def save_parameters(crop_coords):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Crop Coordinate", "Block Size", "C", "Blur Kernel Size", "Morph Kernel Size", "Threshold Type"])
        # Write the parameters
        writer.writerow([crop_coords, block_size, C, blur_kernel_size, morph_kernel_size, threshold_type])

# Mouse callback function for drawing the crop rectangle
def draw_crop_rectangle(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, crop_coordinates

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        crop_coordinates = (start_x, start_y, end_x, end_y)
        # Save new coordinates to CSV
        save_parameters(crop_coordinates)

# Initialize webcam video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_crop_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the previously saved crop rectangle on startup
    if crop_coordinates:
        cv2.rectangle(frame, (crop_coordinates[0], crop_coordinates[1]), (crop_coordinates[2], crop_coordinates[3]), (0, 255, 0), 2)

    # If drawing a new rectangle, draw the current rectangle
    if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1 and drawing:
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

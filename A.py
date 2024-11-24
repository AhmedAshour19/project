import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from datetime import datetime
from ultralytics import YOLO
import streamlit as st

# Define paths to save cropped license plate images and processed video
image_save_path = r"C:\Users\makany\Desktop\project\cropped_plates"
processed_video_path = r"C:\Users\makany\Desktop\project\processed_video.avi"
excel_file_path = r"C:\Users\makany\Desktop\project/car_plate_data.xlsx"


# Define a custom function to draw corner rectangles
def draw_corner_rect(img, bbox, color=(0, 255, 0), thickness=2, length=25):
    x, y, w, h = bbox
    # Top-left corner
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    # Top-right corner
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom-left corner
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom-right corner
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

# Ensure the directory exists
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# Initialize an empty list to store data records for Excel
data_records = []
frame_placeholder = st.empty()

# Load the YOLO models
plate_model = YOLO("best.pt")  # Model for license plate detection
model = YOLO("yolov8n.pt")  # Model for general object detection

# Initialize PaddleOCR for license plate recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Streamlit UI
st.title("Traffic Violation Detection System")

# File uploader for video input
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# Button to start detection
if uploaded_file is not None:
    # Load the video from uploaded file
    cap = cv2.VideoCapture(uploaded_file.name)

    # Define the region of interest (ROI) for the traffic light
    traffic_light_roi = (900, 100, 960, 160)  # Example values

    # Track recognized plates to avoid duplicate records
    processed_numbers = set()
    Traffic_color = ""
    po1_rec = (570, 370)
    po12_rec = (176, 370)

    width = 1020
    height = 720

    # Set up the video writer to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (width, height))

    # Loop through video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame to reduce memory usage
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        # Resize the frame for consistency
        frame = cv2.resize(frame, (width, height))

        # Detect objects (e.g., vehicles) using the general YOLO model
        results = model(frame, stream=True)

        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.line(frame, (po1_rec), (po12_rec), (255, 0, 0), 2)

        # Process the detection results from the general YOLO model
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, z = int(x1), int(y1), int(x2), int(y2)
                w = w - x
                h = z - y
                cx = x + w // 2
                cy = y + h // 2
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Detect vehicles (adjust class indices as per your model)
                if cls in [2,9]:  # Example classes: car, motorcycle, bus, truck, etc.
                    # Draw bounding box using the custom function
                    #draw_corner_rect(frame, (x, y, w, h), color=(0, 255, 0), thickness=2, length=25)
                    if cls==9:
                        draw_corner_rect(frame, (x, y, w, h), color=(0, 255, 0), thickness=2, length=25)
                bitwise = cv2.bitwise_and(frame, black_image)
                gray_image = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)

                # Thresholding
                _, canny_black_image = cv2.threshold(gray_image, 70, 250, cv2.THRESH_BINARY)

                if cv2.countNonZero(canny_black_image[y:y + h // 3, x:x + w]) > 20:
                    Traffic_color = "Red"
                elif cv2.countNonZero(canny_black_image[y + h // 3:y + (h // 3) * 2, x:x + w]) > 20:
                    Traffic_color = "Yellow"
                elif cv2.countNonZero(canny_black_image[y + (h // 3) * 2:y + h, x:x + w]) > 20:
                    Traffic_color = "Green"

                
                    # Detect vehicles (adjust class indices as per your model)
                    if cls in [2,9]:  # Example classes: car, motorcycle, bus, truck, etc.
                        # Draw bounding box using the custom function
                        #draw_corner_rect(frame, (x, y, w, h), color=(0, 255, 0), thickness=2, length=25)
                        if cls==9:
                            draw_corner_rect(frame, (x, y, w, h), color=(0, 255, 0), thickness=2, length=25)
                            

                        # Fill black_image with white rectangle where vehicles are detected
                        cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

                        bitwise = cv2.bitwise_and(frame, black_image)
                        gray_image = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)

                        # Thresholding
                        _, canny_black_image = cv2.threshold(gray_image, 70, 250, cv2.THRESH_BINARY)

                        # Determine Traffic Light Color
                        if cv2.countNonZero(canny_black_image[y:y + h // 3, x:x + w]) > 20:
                            Traffic_color = "Red"
                        elif cv2.countNonZero(canny_black_image[y + h // 3:y + (h // 3) * 2, x:x + w]) > 20:
                            Traffic_color = "Yellow"
                        elif cv2.countNonZero(canny_black_image[y + (h // 3) * 2:y + h, x:x + w]) > 20:
                            Traffic_color = "Green"
                        else:
                            Traffic_color = "Unknown"

                        # Display Traffic Light Color
                        cv2.putText(frame, f"Traffic Light: {Traffic_color}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                        # Check for Red light violation
                        if Traffic_color == "Red" and 452 < cy < 455 and 90 < cx < 800:
                            # Detect license plates using plate_model
                            vehicle_crop = frame[y:y + h, x:x + w]
                             # Draw bounding box using the custom function
                            draw_corner_rect(frame, (x, y, w, h), color=(0, 255, 0), thickness=2, length=25)

                            # Detect license plates
                            plate_results = plate_model(vehicle_crop, stream=True)

                            for plate_result in plate_results:
                                plate_boxes = plate_result.boxes

                                for plate_box in plate_boxes:
                                    px1, py1, px2, py2 = plate_box.xyxy[0]

                                    # Crop the detected license plate
                                    plate_crop = vehicle_crop[int(py1):int(py2), int(px1):int(px2)]

                            # Apply OCR to the cropped plate
                            ocr_result = ocr.ocr(plate_crop, cls=True)
                            if ocr_result and len(ocr_result) > 0:
                                text = ocr_result[0][0][1][0].strip()

                                # Clean the text
                                if text:
                                    text_clean = text.replace(" ", "").strip()
                                else:
                                    text_clean = ""
                                
                                # If valid and not processed before, save it
                                if text_clean and len(text_clean) > 4 and text_clean not in processed_numbers:
                                    processed_numbers.add(text_clean)  # Track the recognized plate
                                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    # Save the cropped image of the license plate
                                    image_filename = f"{text_clean}_{current_datetime.replace(':', '-')}.jpg"
                                    image_filepath = os.path.join(image_save_path, image_filename)
                                    cv2.imwrite(image_filepath, plate_crop)  # Save the image

                                    # Append the record to the data list
                                    data_records.append({
                                        "License Plate": text_clean,
                                        "DateTime": current_datetime,
                                        "Violation": "Red light violation",
                                        "ImagePath": image_filepath
                                    })
        
        # Display every 10th frame in Streamlit to reduce memory consumption
        if frame_count % 10 == 0:
            display_frame = cv2.resize(frame, (width // 2, height // 2))  # Resize for lower memory usage
            frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Write the frame to the processed video
        out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()

    # After processing, save the records to the Excel sheet and display in table format
    if data_records:
        # Load existing data from the Excel file
        if os.path.exists(excel_file_path):
            df_existing = pd.read_excel(excel_file_path)
        else:
            df_existing = pd.DataFrame(columns=["License Plate", "DateTime", "Violation", "ImagePath"])

        # Combine existing data with new data and save to Excel
        df_new = pd.DataFrame(data_records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(excel_file_path, index=False)
        st.success("Data successfully saved to Excel file.")
        st.write("Violation Records:")
        st.dataframe(df_combined[["License Plate", "DateTime", "Violation"]])
    else:
        st.write("No violations detected.")

    # Display the processed video in Streamlit
   # if os.path.exists(processed_video_path):
     #   st.video(processed_video_path)

# Function to search for a license plate number
search_plate = st.text_input("Search for a license plate number:")
if search_plate:
    if os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
        # Search for matching license plate records
        matching_records = df[df['License Plate'].str.contains(search_plate, case=False)]
        
        if not matching_records.empty:
            st.write(f"Records for license plate '{search_plate}':")
            st.dataframe(matching_records[["License Plate", "DateTime", "Violation"]])

            # Loop through the records and display the photos if available
            for _, row in matching_records.iterrows():
                st.write(f"Violation at {row['DateTime']}")
                img_path = row['ImagePath']  # Assuming 'ImagePath' column contains the path to the photo

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=f"License Plate: {row['License Plate']}", use_column_width=True)
                else:
                    st.warning(f"Image not found for this violation: {img_path}")
        else:
            st.write(f"No records found for license plate '{search_plate}'.")

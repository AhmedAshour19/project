import streamlit as st
import cv2
import numpy as np
# import cvzone  # Not used in this version
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import os
import tempfile

import requests





# Telegram Bot Token and Chat ID
TOKEN = "7581663246:AAFNYv6F2JRjPkB-5VtIIIO-zue2SVYysAA"

chat_id = "1347052266"

image_path = "violation_licence.jpg"
image_path1 = "violation_car.jpg"


def Send_Image(bot_token, chat_id, image):
    """
    Sends an image to a Telegram chat.

    Parameters:
    - bot_token: The token for the Telegram bot.
    - chat_id: The ID of the Telegram chat to send the image to.
    - image: The file path of the image to send.
    """

    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    files = {'photo': open(image, 'rb')}
    data = {'chat_id': chat_id}
    response = requests.post(url, files=files, data=data)
    return response.json()

def send_image_and_treatment(annotated_image_path, clean_text):
    """
    Send image and treatment information to Telegram.

    Parameters:
    - annotated_image_path: The path to the annotated image.
    - detected_diseases: A list of detected diseases and their treatments.
    """
    Send_Image(TOKEN, chat_id, annotated_image_path)
    
    message = f"this number: {clean_text}"
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    data = {'chat_id': chat_id, 'text': message}
    requests.post(url, data=data)







# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load YOLO models with caching to improve performance
@st.cache_resource
def load_models():
    plate_model = YOLO("best.pt")  # Ensure 'best.pt' is in the correct path
    general_model = YOLO("yolov8n.pt")  # Ensure 'yolov8n.pt' is in the correct path
    return plate_model, general_model

plate_model, general_model = load_models()

# Define paths for saving data
IMAGE_SAVE_PATH = "saved_images"
PROCESSED_VIDEO_PATH = "processed_video.avi"
DATA_RECORDS_PATH = "car_plate_data.txt"

# Create directories if they don't exist
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

# Initialize data records list
data_records = []

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

# Streamlit Interface
st.title("Red Light Violation Detection System")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Button to start detection
    if st.button("Start Detection"):
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)

        # Define the region of interest (ROI) for the traffic light
        traffic_light_roi = (900, 100, 960, 160)  # Adjust as needed

        # Track recognized plates to avoid duplicate records
        processed_numbers = set()
        Traffic_color = ""
        po1_rec = (570, 370)
        po12_rec = (176, 370)

        # Define video properties
        width = 1020
        height = 720

        # Set up the video writer to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(PROCESSED_VIDEO_PATH, fourcc, 20.0, (width, height))

        # Create a placeholder for video frames
        frame_placeholder = st.empty()

        st.write("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame for consistency
            frame = cv2.resize(frame, (width, height))

            # Detect objects (e.g., vehicles) using the general YOLO model
            results = general_model(frame, stream=True)

            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.line(frame, po1_rec, po12_rec, (255, 0, 0), 2)

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
                        if Traffic_color == "Red" and 452 < cy < 455 and 176 < cx < 570:
                            # Detect license plates using plate_model
                            vehicle_crop = frame[y:y + h, x:x + w]
                            cv2.imwrite(image_path1,vehicle_crop )
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
                                    cv2.imwrite(image_path,plate_crop)

                                    # Apply OCR to the cropped plate
                                    ocr_result = ocr.ocr(plate_crop, cls=True)
                                    if ocr_result and len(ocr_result) > 0:
                                        text = ocr_result[0][0][1][0].strip()

                                        # Clean the text
                                        if text:
                                            text_clean = ''.join(e for e in text if e.isalnum())
                                        else:
                                            text_clean = ""

                                        # If valid and not processed before, save it
                                        if text_clean and len(text_clean) > 4 and text_clean not in processed_numbers:
                                            processed_numbers.add(text_clean)  # Track the recognized plate
                                            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                            # Save the cropped image of the license plate
                                            image_filename = f"{text_clean}_{current_datetime.replace(':', '-')}.jpg"
                                            image_filepath = os.path.join(IMAGE_SAVE_PATH, image_filename)
                                            cv2.imwrite(image_filepath, plate_crop)  # Save the image

                                            # Append the record to the data list
                                            data_records.append({
                                                "License Plate": text_clean,
                                                "DateTime": current_datetime,
                                                "Violation": "Red light violation",
                                                "ImagePath": image_filepath
                                            })

                                            # Optionally, display the recognized plate
                                            cv2.putText(frame, f"Plate: {text_clean}", (x, y - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                            Send_Image(TOKEN,chat_id,image_path1)
                                            send_image_and_treatment(image_path, text_clean)

                                            # Draw a rectangle around the detected license plate
                                            cv2.rectangle(frame, (x + int(px1), y + int(py1)),
                                                          (x + int(px2), y + int(py2)), (0, 255, 0), 2)

            # Display the frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Write the frame to the processed video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()

        st.success("Video processing completed.")

        # Provide a download link for the processed video
        with open(PROCESSED_VIDEO_PATH, "rb") as video_file:
            st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_video.avi",
                mime="video/avi"
            )

        # Display the recorded data
        if data_records:
            st.subheader("Detected Violations")
            for record in data_records:
                st.write(f"**License Plate**: {record['License Plate']}")
                st.write(f"**DateTime**: {record['DateTime']}")
                st.write(f"**Violation**: {record['Violation']}")
                st.write(f"**Image Path**: {record['ImagePath']}")
                st.image(record['ImagePath'], caption=record['License Plate'], use_column_width=True)
                st.markdown("---")

        # Optionally, save data records to a file
        with open(DATA_RECORDS_PATH, "w") as file:
            for record in data_records:
                file.write(f"{record['License Plate']}\t{record['DateTime']}\t{record['Violation']}\t{record['ImagePath']}\n")

        # Provide a download link for the data records
        with open(DATA_RECORDS_PATH, "rb") as data_file:
            st.download_button(
                label="Download Data Records",
                data=data_file,
                file_name="car_plate_data.txt",
                mime="text/plain"
            )

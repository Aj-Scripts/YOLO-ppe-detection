import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
from PIL import Image
import torch

# Simple username and password for authentication
USERNAME = "admin"
PASSWORD = "password"

# Streamlit App
st.set_page_config(page_title="Object Detection App", layout="wide")
st.sidebar.title("Settings")
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #2e2e2e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Login Functionality
def login(username, password):
    return username == USERNAME and password == PASSWORD

# Login Form
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login(username_input, password_input):
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")
else:
    # Select YOLO version
    model_version = st.selectbox("Select YOLO Version", ("YOLOv8", "YOLOv5"))
    
    # Allow the user to upload a model file
    model_file = st.file_uploader(f"Upload {model_version} Model (.pt file)", type=["pt"])
    
    if model_file is not None:
        # Save the uploaded model file
        model_path = 'D:/yolov5safetyhelmet-main/best.pt'
        with open(model_path, 'wb') as f:
            f.write(model_file.getbuffer())

        # Load the custom model
        if model_version == "YOLOv8":
            model = YOLO(model_path)  # Load the YOLOv8 model
        else:  # YOLOv5
            model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)  # Load the YOLOv5 model

        st.write(f"{model_version} model loaded successfully!")

        # List of classes available in your model
        classes = ['helmet', 'person', 'vehicle']  # Example classes
        selected_classes = st.multiselect("Select Classes to Detect", classes, default=classes)

        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["Webcam Detection", "File Processing"])

        with tab1:
            # Button to start detection on webcam
            if st.button("Start Webcam Detection", key="start_webcam"):
                st.session_state.webcam_detection = True

            # Initialize session state for controlling video detection
            if 'webcam_detection' not in st.session_state:
                st.session_state.webcam_detection = False

            # If webcam detection is active
            if st.session_state.webcam_detection:
                video_capture = cv2.VideoCapture(0)  # Change index if needed
                frame_placeholder = st.empty()  # Placeholder for video frames

                while True:
                    ret, frame = video_capture.read()  # Capture a frame from the webcam
                    if not ret:
                        st.write("Failed to capture image")
                        break

                    # Perform inference with selected classes
                    if model_version == "YOLOv8":
                        results = model(frame, classes=[classes.index(cls) for cls in selected_classes])  
                    else:  # YOLOv5
                        results = model(frame)  # YOLOv5 does not take class filters in the same way

                    # Initialize compliance metrics
                    compliance_count = 0

                    # Render results
                    for result in results:
                        if model_version == "YOLOv8":
                            if result.boxes is not None and len(result.boxes) > 0:
                                for box in result.boxes.xyxy:
                                    x1, y1, x2, y2 = box
                                    cls_idx = int(result.boxes.cls[0])  # Get the class index
                                    confidence = result.boxes.conf[0]  # Get the confidence score
                                    
                                    # Check if the class index is valid
                                    if cls_idx < len(result.names):
                                        label = f"{result.names[cls_idx]} {confidence:.2f}"
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                        # Check for compliance
                                        if result.names[cls_idx] == "helmet":
                                            compliance_count += 1
                        else:  # YOLOv5
                            for box in result.xyxy[0]:  # YOLOv5 output format
                                x1, y1, x2, y2, conf, cls_idx = box
                                if cls_idx in selected_classes:  # Check if the class is selected
                                    label = f"{model.names[int(cls_idx)]} {conf:.2f}"
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                    # Check for compliance
                                    if model.names[int(cls_idx)] == "helmet":
                                        compliance_count += 1

                    # Display compliance alerts
                    if compliance_count > 0:
                        st.success("Helmet detected! Compliance requirements met.")

                    # Convert the BGR frame to RGB format
                    frame_with_results = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the results in the Streamlit app
                    frame_placeholder.image(frame_with_results, channels="RGB", use_column_width=True)

                    # Add a button to stop webcam detection with a unique key
                    if st.button("Stop Webcam Detection", key="stop_webcam_unique"):
                        st.session_state.webcam_detection = False
                        break

                # Release the video capture
                video_capture.release()
                st.write("Webcam detection stopped.")

        with tab2:
            # Option to upload a video file
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
            
            # Option to upload an image file
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            # If a file is uploaded, process it
            if uploaded_file is not None:
                # Create a temporary file to save the processed video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                    processed_video_path = temp_video.name
                    video_capture = cv2.VideoCapture(uploaded_file.name)
                    frame_placeholder = st.empty()  # Placeholder for video frames
                    
                    # Create VideoWriter object to save the processed video
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

                    while True:
                        ret, frame = video_capture.read()  # Capture a frame from the uploaded video
                        if not ret:
                            st.write("End of video.")
                            break

                        # Perform inference with selected classes
                        if model_version == "YOLOv8":
                            results = model(frame, classes=[classes.index(cls) for cls in selected_classes])  
                        else:  # YOLOv5
                            results = model(frame)

                        # Initialize compliance metrics
                        compliance_count = 0

                        for result in results:
                            if model_version == "YOLOv8":
                                if result.boxes is not None and len(result.boxes) > 0:
                                    for box in result.boxes.xyxy:
                                        x1, y1, x2, y2 = box
                                        cls_idx = int(result.boxes.cls[0])  # Get the class index
                                        confidence = result.boxes.conf[0]  # Get the confidence score
                                        
                                        # Check if the class index is valid
                                        if cls_idx < len(result.names):
                                            label = f"{result.names[cls_idx]} {confidence:.2f}"
                                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                            # Check for compliance
                                            if result.names[cls_idx] == "helmet":
                                                compliance_count += 1
                            else:  # YOLOv5
                                for box in result.xyxy[0]:  # YOLOv5 output format
                                    x1, y1, x2, y2, conf, cls_idx = box
                                    if cls_idx in selected_classes:  # Check if the class is selected
                                        label = f"{model.names[int(cls_idx)]} {conf:.2f}"
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                        # Check for compliance
                                        if model.names[int(cls_idx)] == "helmet":
                                            compliance_count += 1

                        # Display compliance alerts
                        if compliance_count > 0:
                            st.success("Helmet detected! Compliance requirements met.")

                        # Write the processed frame to the video file
                        out.write(frame)

                        # Convert the BGR frame to RGB format for display
                        frame_with_results = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Display the results in the Streamlit app
                        frame_placeholder.image(frame_with_results, channels="RGB", use_column_width=True)

                    # Release the video capture and writer
                    video_capture.release()
                    out.release()
                    st.write("Video processing complete. You can download the processed video.")

                    # Provide a download button for the processed video
                    with open(processed_video_path, 'rb') as f:
                        st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

            # If an image file is uploaded, process it
            if uploaded_image is not None:
                # Read the image file
                image = Image.open(uploaded_image)
                image = np.array(image)  # Convert to a NumPy array

                # Perform inference with selected classes
                if model_version == "YOLOv8":
                    results = model(image, classes=[classes.index(cls) for cls in selected_classes])  
                else:  # YOLOv5
                    results = model(image)

                # Initialize compliance metrics
                compliance_count = 0

                # Render results
                if model_version == "YOLOv8":
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes.xyxy:
                                x1, y1, x2, y2 = box
                                cls_idx = int(result.boxes.cls[0])  # Get the class index
                                confidence = result.boxes.conf[0]  # Get the confidence score
                                
                                # Check if the class index is valid
                                if cls_idx < len(result.names):
                                    label = f"{result.names[cls_idx]} {confidence:.2f}"
                                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                    # Check for compliance
                                    if result.names[cls_idx] == "helmet":
                                        compliance_count += 1
                else:  # YOLOv5
                    for box in results.xyxy[0]:  # YOLOv5 output format
                        x1, y1, x2, y2, conf, cls_idx = box
                        if cls_idx in selected_classes:  # Check if the class is selected
                            label = f"{model.names[int(cls_idx)]} {conf:.2f}"
                            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            # Check for compliance
                            if model.names[int(cls_idx)] == "helmet":
                                compliance_count += 1

                # Display compliance alerts
                if compliance_count > 0:
                    st.success("Helmet detected! Compliance requirements met.")

                # Convert the BGR frame to RGB format for display
                image_with_results = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Display the results in the Streamlit app
                st.image(image_with_results, channels="RGB", use_column_width=True)

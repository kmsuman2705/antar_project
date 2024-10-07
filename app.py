<<<<<<< HEAD
import os
import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
from io import BytesIO
import pygame
import time
import base64
import pyttsx3
import easyocr
from googletrans import Translator
import tempfile

# audio playback
pygame.mixer.init()

# OCR reader
reader = easyocr.Reader(['en', 'hi',])

# it is  YOLO model
net = cv2.dnn.readNet('file//yolov3.weights', 'file//yolov3.cfg')
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# CoCo class
with open('file//coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

engine = pyttsx3.init()
translator = Translator()

# calculating the distance in meters
def calculate_distance(width, known_width=30, focal_length=700):
    distance = (known_width * focal_length) / width
    return distance

# objects detect
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, 0.5, 0.4)
    results = []
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = calculate_distance(w)
            results.append({
                "label": label,
                "confidence": confidences[i],
                "box": (x, y, w, h),
                "distance": distance
            })

    return results

# detect text using OCR
def detect_text(frame, language):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    if len(result) > 0:
        for (bbox, text, prob) in result:
            if prob > 0.5 and text.strip():
                cv2.rectangle(frame, (int(bbox[0][0]), int(
                    bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(bbox[0][0]), int(
                    bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                speak(text, lang=language[1])

    return frame

# process frame
def process_frame(frame, language):
    detections = detect_objects(frame)

    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        distance = detection["distance"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if language[1] == 'hi':
            translated_label = translator.translate(
                f'{label} detected at {distance / 100:.2f} meters', dest='hi').text
        else:
            translated_label = f'{label} detected at {distance / 100:.2f} meters'

        cv2.putText(frame, translated_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        speak(translated_label, lang=language[1])

    frame = detect_text(frame, language)
    return frame

# speak text
def speak(text, lang='en'):
    if text and text.strip():
        try:
            tts = gTTS(text=text, lang=lang)
            audio_data = BytesIO()
            tts.write_to_fp(audio_data)
            audio_data.seek(0)
            pygame.mixer.music.load(audio_data, "mp3")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except AssertionError:
            print("No valid text to send to TTS API.")
    else:
        print("No valid text to speak.")


def main():
    st.set_page_config(
        page_title="Smart Guide with Text Detection", layout="wide")

    
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_base64 = get_base64_image("file/ist.jpg")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            font-family: 'Arial', sans-serif;
        }}
        .card-box {{
            background-color: rgb(27 47 224 / 10%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgb(0 0 0 / 50%);
            max-width: 700px;
            margin: 50px auto;
            text-align: center;
        }}
        .stButton > button {{
            width: 30%;
            padding: 15px;
            border-radius: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            font-size: 20px;
            margin: 0 auto;
            display: block;
        }}
        .stButton > button:hover {{
            background-color: #0056b3;
        }}
        h1 {{
            color: #fff;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #fff;
            font-size: 2rem;
            margin-bottom: 20px;
        }}
        .center {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image("file/Smart.png", width=150)

    st.markdown(
        """
        <div class="card-box">
            <h1>Smart guide for blind person</h1>
            <h2>Real-Time Object and Text Detection</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False

    language = st.selectbox(
        "Select Language",
        [
            ("English", "en"),
            ("Bengali", "bn"),
            ("Hindi", "hi"),
            ("Tamil", "ta"),
            ("Telugu", "te"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
            ("Korean", "ko")
        ]
    )

    video_source = st.selectbox(
        "Select Frame Source", ["Webcam","Upload Image","Upload Video"])

    toggle = st.button("Start/Stop Stream")

    if toggle:
        st.session_state.is_streaming = not st.session_state.is_streaming
        if st.session_state.is_streaming:
            speak("stream started", lang=language[1])
        else:
            speak("stream stopped", lang=language[1])

    if video_source == "Webcam" and st.session_state.is_streaming:
        speak("You selected webcam", lang=language[1])
    elif video_source == "Upload Image" and st.session_state.is_streaming:
        speak("You selected upload image", lang=language[1])
    elif video_source == "Upload Video" and st.session_state.is_streaming:
        speak("You selected upload video", lang=language[1])
    
    if st.session_state.is_streaming:
        if video_source == "Webcam":
            run_webcam(language)
        elif video_source == "Upload Image":
            uploaded_image = st.file_uploader(
                "Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                process_uploaded_image(uploaded_image, language)
        elif video_source == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload Video", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                process_uploaded_video(uploaded_file, language)
        
# run webcam 
def run_webcam(language):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened() and st.session_state.is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, language)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

#  process uploaded video 
def process_uploaded_video(uploaded_file, language):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filepath = temp_file.name

    cap = cv2.VideoCapture(temp_filepath)

    stframe = st.empty()  

    while cap.isOpened() and st.session_state.is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, language)

        frame_height, frame_width = frame.shape[:2]
        display_width = 640  
        display_height = int(frame_height * (display_width / frame_width))
        frame_resized = cv2.resize(frame, (display_width, display_height))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        stframe.image(frame_rgb, channels='RGB', use_column_width=True)

        time.sleep(0.03)

    cap.release()
    os.remove(temp_filepath)

# uploaded images
def process_uploaded_image(uploaded_image, language):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    frame = process_frame(img, language)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame_rgb, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
=======
import os
import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
from io import BytesIO
import pygame
import time
import base64
import pyttsx3
import easyocr
from googletrans import Translator
import tempfile

# audio playback
pygame.mixer.init()

# OCR reader
reader = easyocr.Reader(['en', 'hi',])

# it is  YOLO model
net = cv2.dnn.readNet('file//yolov3.weights', 'file//yolov3.cfg')
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# CoCo class
with open('file//coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

engine = pyttsx3.init()
translator = Translator()

# calculating the distance in meters
def calculate_distance(width, known_width=30, focal_length=700):
    distance = (known_width * focal_length) / width
    return distance

# objects detect
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, 0.5, 0.4)
    results = []
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = calculate_distance(w)
            results.append({
                "label": label,
                "confidence": confidences[i],
                "box": (x, y, w, h),
                "distance": distance
            })

    return results

# detect text using OCR
def detect_text(frame, language):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    if len(result) > 0:
        for (bbox, text, prob) in result:
            if prob > 0.5 and text.strip():
                cv2.rectangle(frame, (int(bbox[0][0]), int(
                    bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(bbox[0][0]), int(
                    bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                speak(text, lang=language[1])

    return frame

# process frame
def process_frame(frame, language):
    detections = detect_objects(frame)

    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        distance = detection["distance"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if language[1] == 'hi':
            translated_label = translator.translate(
                f'{label} detected at {distance / 100:.2f} meters', dest='hi').text
        else:
            translated_label = f'{label} detected at {distance / 100:.2f} meters'

        cv2.putText(frame, translated_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        speak(translated_label, lang=language[1])

    frame = detect_text(frame, language)
    return frame

# speak text
def speak(text, lang='en'):
    if text and text.strip():
        try:
            tts = gTTS(text=text, lang=lang)
            audio_data = BytesIO()
            tts.write_to_fp(audio_data)
            audio_data.seek(0)
            pygame.mixer.music.load(audio_data, "mp3")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except AssertionError:
            print("No valid text to send to TTS API.")
    else:
        print("No valid text to speak.")


def main():
    st.set_page_config(
        page_title="Smart Guide with Text Detection", layout="wide")

    
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_base64 = get_base64_image("file/ist.jpg")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            font-family: 'Arial', sans-serif;
        }}
        .card-box {{
            background-color: rgb(27 47 224 / 10%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgb(0 0 0 / 50%);
            max-width: 700px;
            margin: 50px auto;
            text-align: center;
        }}
        .stButton > button {{
            width: 30%;
            padding: 15px;
            border-radius: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            font-size: 20px;
            margin: 0 auto;
            display: block;
        }}
        .stButton > button:hover {{
            background-color: #0056b3;
        }}
        h1 {{
            color: #fff;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #fff;
            font-size: 2rem;
            margin-bottom: 20px;
        }}
        .center {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image("file/Smart.png", width=150)

    st.markdown(
        """
        <div class="card-box">
            <h1>Smart guide for blind person</h1>
            <h2>Real-Time Object and Text Detection</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False

    language = st.selectbox(
        "Select Language",
        [
            ("English", "en"),
            ("Bengali", "bn"),
            ("Hindi", "hi"),
            ("Tamil", "ta"),
            ("Telugu", "te"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
            ("Korean", "ko")
        ]
    )

    video_source = st.selectbox(
        "Select Frame Source", ["Webcam","Upload Image","Upload Video"])

    toggle = st.button("Start/Stop Stream")

    if toggle:
        st.session_state.is_streaming = not st.session_state.is_streaming
        if st.session_state.is_streaming:
            speak("stream started", lang=language[1])
        else:
            speak("stream stopped", lang=language[1])

    if video_source == "Webcam" and st.session_state.is_streaming:
        speak("You selected webcam", lang=language[1])
    elif video_source == "Upload Image" and st.session_state.is_streaming:
        speak("You selected upload image", lang=language[1])
    elif video_source == "Upload Video" and st.session_state.is_streaming:
        speak("You selected upload video", lang=language[1])
    
    if st.session_state.is_streaming:
        if video_source == "Webcam":
            run_webcam(language)
        elif video_source == "Upload Image":
            uploaded_image = st.file_uploader(
                "Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                process_uploaded_image(uploaded_image, language)
        elif video_source == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload Video", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                process_uploaded_video(uploaded_file, language)
        
# run webcam 
def run_webcam(language):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened() and st.session_state.is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, language)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

#  process uploaded video 
def process_uploaded_video(uploaded_file, language):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filepath = temp_file.name

    cap = cv2.VideoCapture(temp_filepath)

    stframe = st.empty()  

    while cap.isOpened() and st.session_state.is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, language)

        frame_height, frame_width = frame.shape[:2]
        display_width = 640  
        display_height = int(frame_height * (display_width / frame_width))
        frame_resized = cv2.resize(frame, (display_width, display_height))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        stframe.image(frame_rgb, channels='RGB', use_column_width=True)

        time.sleep(0.03)

    cap.release()
    os.remove(temp_filepath)

# uploaded images
def process_uploaded_image(uploaded_image, language):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    frame = process_frame(img, language)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame_rgb, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
>>>>>>> 820fdeb (final)

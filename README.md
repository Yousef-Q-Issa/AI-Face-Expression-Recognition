# AI Face Recognition & Facial Expression Analysis

A Python-based computer vision project for real-time face recognition and facial expression analysis using a webcam or static images.

## Features

- Real-time face detection and recognition
- GPU acceleration using CUDA where supported
- Facial expression recognition (Happy, Sad, Angry, Neutral, Surprise, and more)
- Age estimation and gender classification
- Webcam and image recognition modes
- **Interactive Facial Expression Game** — challenges users to imitate target facial expressions detected by the camera
- Real-time game feedback and scoring
- Multiple-face support
- Automatic age and gender prediction after the game
- Processed image output

## Interactive Facial Expression Game

The project includes an interactive game that challenges users to imitate facial expressions displayed on screen. The webcam analyzes the visible expression and provides feedback and a score based on the model's predictions.

The game is designed as an educational computer vision demonstration and is not intended to measure a person's actual emotional state.

## Technologies

- Python
- OpenCV
- NumPy
- ONNX Runtime
- CUDA
- Pretrained deep learning models

## How It Works

The application detects faces and uses face embeddings to compare detected faces with locally enrolled identities. It also estimates age and gender and classifies visible facial expressions using pretrained models.

People who are not enrolled in the local face database should be classified as Unknown when they do not meet the configured matching threshold.

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Download the required pretrained models using the instructions in `models/README.md`.
5. Connect a webcam or prepare a supported image.
6. Run the application using the instructions below.

## Usage

Run the webcam application:

```bash
python recognize_webcam.py
```

Run image recognition:

```bash
python recognize_image.py
```

Adjust the commands to match the actual file locations in the repository.

## Privacy

Personal face images, face embeddings, and generated recognition outputs are not included in this repository.

Facial expression predictions, age estimates, and gender classifications are model outputs and may be inaccurate. They should not be treated as verified personal attributes or psychological assessments.

## Limitations

Recognition accuracy depends on image quality, lighting, face angle, the enrollment dataset, and the selected matching threshold. GPU acceleration depends on compatible hardware, drivers, and installed runtime providers.

This project is intended for educational and research purposes.

## Screenshots
<img width="635" height="475" alt="Screenshot 2026-10-09 195637" src="https://github.com/user-attachments/assets/ecd24631-08e9-4751-89dc-cc038d17cfba" />
<img width="796" height="648" alt="Screenshot 2026-10-09 195736" src="https://github.com/user-attachments/assets/1f8e64a1-263f-4361-b7d5-a5dacd90b67a" />
<img width="1530" height="790" alt="Screenshot 2026-10-10 170346" src="https://github.com/user-attachments/assets/537c099a-c20b-4349-8440-0ed7a8629cc5" />
<img width="2536" height="1338" alt="image" src="https://github.com/user-attachments/assets/1bb274a9-b50a-49b2-a7f9-7532083ad49f" />





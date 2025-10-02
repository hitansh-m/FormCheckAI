FormChecker AI
Overview

FormChecker AI is a web-based application designed to help users perform exercises safely by analyzing their form through AI-powered pose detection. The app leverages Flask for the backend, MediaPipe for pose detection, and OpenCV for image processing.

What started as a simple plank checker has now grown into a broader project aimed at reducing injuries caused by incorrect exercise form. By providing instant, accessible feedback, FormChecker AI makes proper training guidance available to anyone—not just those with access to personal trainers.

Features

Upload an image of your exercise (currently supports planks, with more exercises in progress).

Analyze posture using AI-powered key point detection.

Visualize form with annotated points and angles.

Receive feedback on whether your form is correct or needs adjustment.

Expanding support to up to 10 different exercises (currently in testing).

Project Structure

app.py – Main Flask application handling image upload, analysis, and response.

index.html – Frontend for user interaction.

Installation

Clone the repository:

git clone https://github.com/hitansh-m/FormChecker-AI
cd FormChecker-AI


Install the required dependencies:

pip install -r requirements.txt


Run the application:

python app.py


Navigate to http://localhost:8888 in your browser.

Usage

Upload an image of your exercise (JPG/PNG).

The app will process the image, detect key points, and analyze posture.

View visual feedback and suggested improvements.

Roadmap

Expand exercise coverage (squats, push-ups, lunges, etc.).

Improve feedback accuracy with angle-based scoring.

Add support for live video analysis.

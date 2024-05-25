# FormCheckAI
Overview
AI Plank Trainer is a web-based application that helps users maintain the correct plank form by analyzing an uploaded image using AI. The application uses Flask for the backend, MediaPipe for pose detection, and OpenCV for image processing.

Features
Upload an image of your plank position.
Analyze the plank form for correctness.
Visualize key points and angles on the uploaded image.
Receive feedback on whether your plank form is correct or not.
Project Structure
app.py: The main Flask application file that handles image upload, analysis, and response.
index.html: The frontend HTML file for user interaction.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/hitansh-m/FormCheckAI
cd ai-plank-trainer
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt (Flask, OpenCV, MediaPipe, NumPy, and OS)
Run the application:

bash
Copy code
python app.py
Open your web browser and navigate to http://localhost:8888.

Usage
Upload Image: On the home page, click on the "Upload Your Plank Image" button and select an image file (JPG, PNG) of your plank position.
Analyze: After uploading, the app will process the image to detect key points and angles.
Feedback: The app will display feedback on whether your plank form is correct or needs improvement.

Contact
For questions or suggestions, please open an issue or contact us via email: hitanshm07@gmail.com

Thank you for using AI Plank Trainer! We hope it helps you achieve the perfect plank form.

🧠 Eye Disease Detection System

A deep learning-based web application that detects eye diseases from retinal images using a combination of CNN and MobileNet models. The system provides real-time predictions through a Flask web interface and is deployed on AWS EC2.

🚀 Features
Upload retinal image and get instant prediction
Ensemble model (CNN + MobileNet) for improved accuracy
Confidence-based prediction fusion
User-friendly web interface
Deployed on AWS EC2 with production setup (Gunicorn + Nginx)

🧠 Model Details
CNN Model
Input: Grayscale images (128×128×1)
Learns structural patterns
MobileNet Model
Input: RGB images (224×224×3)
Learns high-level features
Fusion Technique
Combines predictions based on confidence scores
Improves overall classification performance

🛠️ Tech Stack
Python
TensorFlow / Keras
OpenCV
Flask
AWS EC2
Gunicorn
Nginx

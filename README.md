Abstract:

With the advancement in technology, it is only natural that techniques like Artificial intelligence and machine learning be applied to make life easier. One such instance is this project with Emotion recognition via face detection embedded in a smart recommendation system that suggests music, movies and playlists according to the users mood. A field with a lot of scope and future work possible, this Smart entertainment recommendation system has been built as a web application.
Oftentimes, it so happens that a person just wants to sit back and enjoy some music or watch a movie. Sometimes, said person might not even be aware of what it is exactly that they want, only that their mood is off and that they would really like some entertainment. Sometimes, one might not even know what to watch, and would like to be surprised with a random suggestion. Emote is a web application that does just that. Through smart recognition of emotions, Emote suggests various entertainment choices for the user in a hands-free environment.

INTRODUCTION

Emote is a webapp that recommends music according to the mood of the user using facial detection and emotion recognition. This project was built as a part of the Microsoft Mentor Engage Program and falls under the first problem set of Facial Recognition. This implements a machine learning model for emotion detection that is implemented over the facial detection module.

MOTIVATION

❄️ Nowadays we have so many websites that categorize movies based on their successful ratings, actors who have worked on it, their box office collections and so on and also categorise songs based on the users previou choices.
❄️ People watch movies or listen to songs so that they can relate to the feel of it , to relieve themselves etc..
❄️ But there are hardly any websites which recommend movies/songs based on user's current emotions.
❄️ The proposed entertainment system eliminates the time-consuming and tedious task of manually Segregating or grouping movies/songs into different lists and helps in generating an appropriate movie/song list based on an individual's emotional features.


OBJECTIVES:
To create a web based application that recognises a users mood and recommends entertainment in the form of music, videos and movies.


MOTIVATION
Emote has been developed using many machine learning algorithms including convolution neural networks (CNN) for a facial expression recognition task. The goal is to classify each facial image into one of the seven facial emotion categories considered in this study- Happy, Angry, Sad, Surprised, Disgusted, Neutral, Scared.  CNN models are trained with different depth using gray-scale images from the Kaggle website. This model can be trained in any python development environment. Visualization of different layers of a network is presented to show what features of a face can be learned by CNN models. This trained model is then passed into the main program which identifies the emotion in a live video stream input from the webcam and suggests corresponding entertainment options.
USER CHARACTERISTICS AND EXPECTATIONS:
Users should be internet literate and should be able to navigate the web.
To run application in a development environment, users should have basic knowledge of python, working with importing libraries and basic web development knowledge to parse through the user interface design.

FUNCTIONAL REQUIREMENTS:
Landing Page
Login for the user
Face detection and Emotion Recognition via live stream
Options to choose movie/ music/ playlist

NON-FUNCTIONAL REQUIREMENTS:

1)Usability:
Users should be able to easily navigate the application, find the login icon easily and should also be able to figure out how to navigate to the web live streaming page
Scroll to top must be available from all parts of the page
2) Performance:
Webcam must live stream the video without buffering or disturbances
Emotion recognition must be prompt and must not have a delay of any more than 1ms.
The recommendation must be available immediately after the user chooses a mode of entertainment.
Application must work on and support major browsers like: Google chrome, Firefox and Microsoft Edge

SYSTEM REQUIREMENTS

1)2BASIC REQUIREMENTS
1.1. Stable Internet Connection
1.2. A working web browser supporting JavaScript and Stylesheets.

2)OPERATING SYSTEM
2.1. Microsoft Windows 8/9/10/11
2.2. Mac OS X 10.10 or higher
2.3. iOS 6 or higher
2.4. Android 6 or higher

3. BROWSER REQUIREMENTS
(The latest recommended OS for the relative operating system)
3.1. Firefox 
3.2. Google Chrome 
3.3. Microsoft Edge 

HARDWARE REQUIREMENTS:
Webcam connected to the device

SOFTWARE REQUIREMENTS:

1. HTML
1.1. HyperText Markup Language is the standard markup language used for structuring documents which are to be displayed via a web browser or equivalent software frameworks like Electron.
2. CSS
2.1. Cascading Style Sheets is a styling language for adding styles such as background color, text sizing, fonts to web documents.
3. JavaScript
3.1. JavaScript is a lightweight, interpreted, or just-in-time compiled programming language with first-class functions. It is used to develop the logic for front end applications. Since the inception of nodejs, the language can also be used for server side programming.
4.Python interpreter
5. Development environment that supports python like pycharm, jupyter notebooks, vs code
6.Flask
7.Ajax
8.Bootstrap
9. Git - Version Control System
 Git is a free and open source distributed version control system. It is used for tracking changes in any set of files, usually used for coordinating work among programmers collaboratively developing source code during software development.
10.GitHub
GitHub code hosting platform for software development and version control using Git. It offers the distributed version control and source code management functionality of Git, plus its own features.

**IMPLEMENTATION/ RUNNING THE PROGRAM:**

TRAINING THE EMOTION RECOGNITION MODEL:
This can be done by navigating to the “Model” directory and running emorecog.py.
The pre-run output of the training is available in model_csv.h5 in the same directory.

This output will be passed into the main program- emotion_detection.py
Emotion_detection.py holds the program for the flask template and also for accessing the webcam, classifying the emotions and displaying the corresponding output.
This is the main file that requires to be run for the recommendation system web app.
This requires the user to import various python modules and install dependencies accordingly.
If working in an environment like pycharm, install the required libraries in the local interpreter and import the modules.
This can be generally done by the command: pip install -r requirements.txt

If there is still some error thrown by virtue of the modules not being recognised, import the libraries manually with “pip install”
Run the python file and navigate to http://127.0.0.1:5000 on the browser to open the flask user interface.
Before running the program make sure that all the local paths have been changed to the path on your device.

Installations
Follow steps to use this project:
1)Clone repository
git clone https://github.com/593c723/Emotion-Detection.git
2)Change directory to clone repository
cd Emotion-Detection
3)Create a Python virtual environment and activate it
$ virtualenv venv
$ source venv/bin/activate      # For Linux
$ venv\Scripts\activate         # For Windows
4)Install required libraries
pip install -r requirements.txt
5)Getting Started
Change directory and Run File
cd Flask
python emotion_detection.py

FEATURES:
Emotional Detection Model use Deep Neural Network with CNN architecture model for Image Classification.
Tensorflow-Keras Convolutional Neural Network with multiple layers is used to train and test model for seven classes - Happy, Angry, Neutral, Sad, Surprise, Fear and Disgust.
Emotional Detection Model integrated with OpenCV to capture user facial expression.
Recommendations based on detected mood is done via 3 functions.
Engaging UI is developed using Flask Web Application

DESIGN:
LANDING PAGE:
 
 
LOGIN:
 


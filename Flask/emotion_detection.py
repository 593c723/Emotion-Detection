from flask import Flask, render_template, Response, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image
import webbrowser
import cv2
import numpy as np
from scipy import stats
import time
import os, random
import subprocess
import re
import requests as HTTP
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier('../Model/haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('../Model/model_csv.h5')
label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}

global capture
capture=0
url = 'https://www.youtube.com/results?search_query='
chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
playlist = ''

now = time.time()


def gen_frames():
    global capture
    count = 0
    final_pred = []
    while True:
        success, frame = camera.read()
        cap_img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(cap_img_gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(255,255,255),2)
            count +=1
            roi_gray = cap_img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = tf.keras.utils.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

            predictions = model.predict(roi)[0]
            emotion_label = np.argmax(predictions)
            emotion_prediction = label_dict[emotion_label]
            cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,0), 1 )
            print(emotion_prediction)
            final_pred.append(emotion_prediction)
            if count>=30:
                break
            test_1 = np.array(final_pred)
            mode = stats.mode(test_1)

        if success: 
            if(capture):
                capture=0
                if time.time():

                    def music():
                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        if mode[0][0]== 'Neutral' :
                            randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)

                        elif mode[0][0] == 'Happiness':
                            randomfile = random.choice(
                                os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/happy/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/happy/" + randomfile)

                        elif mode[0][0] == 'Angry':
                            randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/angry/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/angry/" + randomfile)
                        elif mode[0][0] == 'Sad':
                            randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/sad/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/sad/" + randomfile)
                        elif mode[0][0] == 'Disgust':
                            randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/happy/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/happy/" + randomfile)
                        elif mode[0][0] == 'Fear':
                            randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        else:
                            # Surprise: film noir
                            randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/surprise/"))
                            file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/surprise/" + randomfile)


                        subprocess.call([mp, file])
                    def play():
                        if mode[0][0] == 'Neutral':
                            playlist = 'neutral+songs playlist'
                        elif mode[0][0] == 'Happiness':
                            playlist = 'Happy+songs playlist'
                        elif mode[0][0] == 'Angry':
                            playlist = 'Angry+songs playlist'
                        elif mode[0][0] == 'Sad':
                            playlist = 'Sad+songs playlist'
                        elif mode[0][0] == 'Disgust':
                            playlist = 'Cute+songs playlist'
                        elif mode[0][0] == 'Fear':
                            playlist = 'Scary+songs playlist'
                        else:
                            playlist = 'soothing+songs playlist'

                        webbrowser.get(chrome_path).open(url + playlist)

                    def movie():
                        if mode[0][0] == 'Neutral':
                            urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'

                        elif mode[0][0] == 'Happiness':
                            urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
                        elif mode[0][0] == 'Angry':
                            urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'
                        elif mode[0][0] == 'Sad':
                            urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
                        elif mode[0][0] == 'Disgust':
                            urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'
                        elif mode[0][0] == 'Fear':
                            urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'
                        else:
                            urlhere = 'http://www.imdb.com/search/title?genres=film_noir&title_type=feature&sort=moviemeter, asc'

                        webbrowser.get(chrome_path).open(urlhere)
############################################################################################################################
                    @app.route('/playlist')
                    def playlist():
                        play()

                    @app.route('/mov')
                    def mov():
                        movie()

                    @app.route('/songs')
                    def songs():
                        music()

            try:
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('doc.html')


@app.route('/face_detection')
def face_detection():
    return render_template('face_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            
    elif request.method=='GET':
        return render_template('face_detection.html')
    return render_template('face_detection.html')

if __name__ == '__main__':
    app.run()

cv2.destroyAllWindows()

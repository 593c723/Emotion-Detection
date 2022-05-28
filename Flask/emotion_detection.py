from flask import Flask, render_template, Response, request
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
future=now+10

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
                if time.time() > future:
                    if mode[0][0] == 'Neutral':
                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])

                        playlist = 'neutral+songs'
                        webbrowser.get(chrome_path).open(url+playlist)
                    elif mode[0][0] == 'Happiness':
                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(
                            os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])

                        playlist = 'happy+songs'
                        webbrowser.get(chrome_path).open(url+playlist)
                    elif mode[0][0] == 'Angry':

                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(
                            os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])


                        playlist = 'angry+songs'
                        webbrowser.get(chrome_path).open(url+playlist)
                    elif mode[0][0] == 'Sad':

                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(
                            os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])

                        playlist = 'sad+songs playlist'
                        webbrowser.get(chrome_path).open(url+playlist)
                    elif mode[0][0] == 'Disgust':
                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(
                            os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])

                        playlist = 'disgust+songs'
                        webbrowser.get(chrome_path).open(url+playlist)
                    elif mode[0][0] == 'Fear':
                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(
                            os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])

                        playlist = 'fear+songs'
                        webbrowser.get(chrome_path).open(url+playlist)
                    else:
                        mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
                        randomfile = random.choice(
                            os.listdir("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/"))
                        print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                        file = ("C:/Users/sumana/PycharmProjects/Emotion-Detection/Flask/songs/neutral/" + randomfile)
                        subprocess.call([mp, file])

                        playlist = 'surprise+songs'
                        webbrowser.get(chrome_path).open(url+playlist)
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

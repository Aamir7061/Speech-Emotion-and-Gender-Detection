from flask import Flask, render_template, send_file, request
import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
import librosa
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')
g_model=load_model('gender.h5')
def start_stream():
    global audio_data
    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.extend(indata.flatten())

    stream = sd.InputStream(callback=callback, channels=1, dtype=np.int16, samplerate=44100)
    stream.start()
    return stream

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    global stream
    global recording

    if not recording:
        recording = True
        stream = start_stream()

    return 'Recording started'

@app.route('/pause_recording')
def pause_recording():
    global stream
    global recording

    if recording:
        stream.stop()
        recording = False

    return 'Recording paused'

@app.route('/stop_recording', methods=['GET', 'POST'])

def stop_recording():
    global stream
    global recording
    global audio_data

    if recording:
        try:
            print("Stop")
            # Stop and close the audio stream
            stream.stop()
            stream.close()

            # Save the recorded audio to a WAV file
            filename = 'temp.wav'
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(np.array(audio_data).tobytes())

            # Perform emotion prediction or other processing here
            print("save")
            # Reset global variables
            recording = False
            audio_data = []

            return 'Recording stopped and saved successfully.'
        except Exception as e:
            return f'Error stopping recording: {str(e)}'

    return 'No recording to stop'


#@app.route('/predict_emotion', methods=['GET','POST'])


@app.route('/get_audio')
def get_audio():
    print("Sent Audio")
    return send_file('temp.wav', as_attachment=True)



@app .route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        
        # def extract_mfcc(filename):
        #     # Load audio file and convert to NumPy array
        #     y, sr = librosa.load(filename, duration=3, offset=0.5)
        #     y = np.asarray(y)

        #     # Apply a more lenient condition for filtering
        #     a = y[(y > 0.0005) | (y < -0.0005)]

        #     # Check if the resulting array is not empty
        #     if not a.any():
        #         return np.zeros(90)  # Return zeros if the filtered array is empty

        #     # Extract MFCCs
        #     mfcc = np.mean(librosa.feature.mfcc(y=a, sr=sr, n_mfcc=90).T, axis=0)
        #     return mfcc

#         def silence_features(h):
#             y, sr = librosa.load(h)
#             yp = np.where(np.abs(y) > 0.0005)[0]
#             mfccs = librosa.feature.mfcc(y=y[yp], sr=sr, n_mfcc=90)
            
#             # Transpose the MFCCs array to have shape (90, duration)
#             mfccs_reshaped = mfccs.T
            
#             return mfccs_reshaped
#         print("JJJJJJ")
#         features = silence_features('temp.wav')
#         print("JJJNANBAB")
#         features = features.T

# # Add a new axis to match the expected input shape (None, 90, 1)
#         features = np.expand_dims(features, axis=-1)
#         print(features.shape)
#         # features = np.expand_dims(features, axis=0)
        
        
        # def silence_features(h, n_mfcc=90):
        #     y, sr = librosa.load(h)
        #     yp = np.where(np.abs(y) > 0.0005)[0]
        #     mfccs = librosa.feature.mfcc(y=y[yp], sr=sr, n_mfcc=n_mfcc)
            
        #     # Transpose the MFCCs array to have shape (duration, n_mfcc)
        #     mfccs = mfccs.T
            
        #     return mfccs
        
        def extract_mfcc(filename):
            y, sr = librosa.load(filename, duration=3, offset=0.65)
            yt, _ = librosa.effects.trim(y, top_db=20)
            if max(yt)<0.003:
                return 0
            else:
                mfcc = np.mean(librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=90).T, axis=0)
                return mfcc
        
        features = extract_mfcc('temp.wav')

        # Example usage:
        # audio_file_path = "temp.wav"
        # features = silence_features(audio_file_path)
        if isinstance(features, int) and features == 0:
            return render_template('index.html',gender_pred="No " , prediction_text="   Audio", confidence="   Detected")

        # Add a new axis to match the expected input shape (None, 90, 1)
        else:

            features = np.expand_dims(features, axis=0)
            
            gender_pr=g_model.predict(features)
            prediction = model.predict(features)
            print(prediction)
            output = np.argmax(prediction)
            sex = np.argmax(gender_pr)
            # Get the corresponding probabilities
            tlp = prediction[0, output]
            tlp=round(tlp*100)
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
            gender = ["Female","Male"]
            result = emotions[output]
            gresult=gender[sex]
            print(result,tlp)
            return render_template('index.html',gender_pred="Gender {}     -".format(gresult) , prediction_text="   Emotion {}    -".format(result), confidence="  Accuracy {} %   ".format(tlp))
    return render_template('index.html')


if __name__ == '__main__':
    recording = False
    stream = None
    app.run(debug=True)

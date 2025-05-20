# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd

from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64



df = pd.read_csv("/home/shreyashi/College_project/Emotion-based-music-recommendation-system/muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
print(df)

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
print(df)

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):

    data = pd.DataFrame()

    # If list is empty, use "Happy" as default
    if not list:
        list = ["Happy"]

    if len(list) == 1:
        v = list[0]
        t = 30
        v_lower = v.lower() if isinstance(v, str) else ""
        if v_lower == 'neutral' or v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v_lower == 'angry' or v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v_lower == 'fear' or v_lower == 'fearful' or v == 'Fearful':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v_lower == 'happy' or v == 'Happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        elif v_lower == 'sad' or v == 'Sad':
            data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
        else:
            # Default to happy if emotion not recognized
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            v_lower = v.lower() if isinstance(v, str) else ""
            if v_lower == 'neutral' or v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v_lower == 'angry' or v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v_lower == 'fear' or v_lower == 'fearful' or v == 'Fearful':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v_lower == 'happy' or v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            elif v_lower == 'sad' or v == 'Sad':
               data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
            else:
               # Default to happy if emotion not recognized
               data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)):
            v = list[i]
            t = times[i]

            v_lower = v.lower() if isinstance(v, str) else ""
            if v_lower == 'neutral' or v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v_lower == 'angry' or v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v_lower == 'fear' or v_lower == 'fearful' or v == 'Fearful':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v_lower == 'happy' or v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            elif v_lower == 'sad' or v == 'Sad':
                data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
            else:
                # Default to happy if emotion not recognized
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            v_lower = v.lower() if isinstance(v, str) else ""
            if v_lower == 'neutral' or v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v_lower == 'angry' or v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v_lower == 'fear' or v_lower == 'fearful' or v == 'Fearful':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v_lower == 'happy' or v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            elif v_lower == 'sad' or v == 'Sad':
                data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
            else:
                # Default to happy if emotion not recognized
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            v_lower = v.lower() if isinstance(v, str) else ""
            if v_lower == 'neutral' or v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v_lower == 'angry' or v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v_lower == 'fear' or v_lower == 'fearful' or v == 'Fearful':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v_lower == 'happy' or v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            elif v_lower == 'sad' or v == 'Sad':
                data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
            else:
                # Default to happy if emotion not recognized
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul





model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.load_weights('/home/shreyashi/College_project/Emotion-based-music-recommendation-system/model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier('/home/shreyashi/College_project/Emotion-based-music-recommendation-system/haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

# Custom CSS for the entire app
custom_css = '''
<style>
    /* Main background and font styles */
    body {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        font-family: 'Poppins', sans-serif;
        color: white;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .main-header p {
        font-size: 1.2rem;
        color: #e0e0e0;
        margin-top: 0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    /* Song recommendation styling */
    .song-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .song-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }

    .song-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .song-title a {
        color: #ffffff;
        text-decoration: none;
        transition: color 0.3s ease;
    }

    .song-title a:hover {
        color: #ff8a00;
    }

    .artist-name {
        font-size: 1rem;
        color: #e0e0e0;
        font-style: italic;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }

    .footer h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }

    .team-members {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1.5rem;
    }

    .team-member {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        font-weight: 500;
        color: white;
        transition: all 0.3s ease;
    }

    a .team-member {
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    a .team-member:hover {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
    }

    /* Status messages */
    .success-message {
        background: linear-gradient(90deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .warning-message {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    /* Section headers */
    .section-header {
        text-align: center;
        margin: 2rem 0 1rem 0;
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Dividers */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        margin: 1.5rem 0;
        border-radius: 3px;
    }
</style>

<!-- Import Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
'''

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Header section
st.markdown('''
<div class="main-header">
    <h1>Emotion Based Music Recommendation</h1>
    <p>Discover music that matches your mood</p>
</div>
''', unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

list = []
# Create a container for the scan button
st.markdown('<div style="display: flex; justify-content: center; margin: 2rem 0;">', unsafe_allow_html=True)
if st.button('SCAN MY EMOTION'):

        count = 0
        list.clear()

        # Check if OpenCV GUI is available
        has_gui = True
        try:
            # Test if we can create a window
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Test")
        except Exception as e:
            print(f"Warning: OpenCV GUI functionality not available: {e}")
            has_gui = False
            # If GUI is not available, we'll still process frames but not display them

        # Check if camera is available
        ret, test_frame = cap.read()
        camera_available = ret

        if not camera_available:
            st.markdown('<div class="warning-message">Camera not available. Using default emotion for recommendations.</div>', unsafe_allow_html=True)
            # Add a default emotion when camera is not available
            list.append("Happy")
            list = pre(list)
            st.markdown('<div class="success-message">Using default emotion: Happy</div>', unsafe_allow_html=True)
        else:
            # Only enter the video capture loop if camera is available
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                count = count + 1

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))

                    list.append(emotion_dict[max_index])

                    cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Only try to display if GUI is available
                    if has_gui:
                        try:
                            cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))
                        except Exception as e:
                            print(f"Warning: Could not display video frame: {e}")
                            has_gui = False  # Disable GUI for future iterations

                # Only try to wait for key if GUI is available
                if has_gui:
                    try:
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            break
                    except Exception as e:
                        print(f"Warning: Error in waitKey: {e}")
                        has_gui = False  # Disable GUI for future iterations
                if count >= 20:
                    break
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Warning: Could not destroy OpenCV windows: {e}")
            # Continue execution even if destroyAllWindows fails

        # Check if any emotions were detected
        if not list:
            st.markdown('<div class="warning-message">No faces detected. Using default emotion for recommendations.</div>', unsafe_allow_html=True)
            # Add a default emotion when no faces are detected
            list.append("Happy")

        list = pre(list)
        st.markdown('<div class="success-message">Emotions successfully detected! Finding your perfect music...</div>', unsafe_allow_html=True)


# Close the button container div
st.markdown('</div>', unsafe_allow_html=True)

new_df = fun(list)

# Add a divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Section header for recommendations
st.markdown('<h2 class="section-header">Your Personalized Music Recommendations</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Click on any song title to listen</p>', unsafe_allow_html=True)

try:
    # Create a container for the song recommendations
    for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):
        st.markdown(f'''
        <div class="song-container">
            <div class="song-title">
                <a href="{l}" target="_blank">{i+1}. {n}</a>
            </div>
            <div class="artist-name">
                {a}
            </div>
        </div>
        ''', unsafe_allow_html=True)
except:
    st.markdown('<p style="text-align: center; padding: 2rem;">No recommendations available. Please try scanning your emotion again.</p>', unsafe_allow_html=True)

# Add footer with team members
st.markdown('''
<div class="footer">
    <h3>Developed by</h3>
    <div class="team-members">
        <a href="https://www.linkedin.com/in/shreyashi36/" target="_blank" style="text-decoration: none;">
            <div class="team-member">Shreyashi Das</div>
        </a>
        <div class="team-member">Puspal Paul</div>
        <div class="team-member">Sentu Naskar</div>
        <div class="team-member">Anwesha Mondal</div>
    </div>
</div>
''', unsafe_allow_html=True)
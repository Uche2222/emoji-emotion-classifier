import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Emoji Emotion Classifier",
    page_icon="😀",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS for Background & Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    }

    /* Center everything */
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Card style for input box and output */
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
        font-size: 24px;
        text-align: center;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Arial', sans-serif;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Dataset & Train Model
# -----------------------------
df = pd.read_csv("ijstable.csv")
df = df[pd.to_numeric(df['Sentiment score'], errors='coerce').notnull()]
df['Sentiment score'] = df['Sentiment score'].astype(float)

def get_emotion(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['Emotion'] = df['Sentiment score'].apply(get_emotion)

X = df['Char']
y = df['Emotion']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

vectorizer = TfidfVectorizer(analyzer='char')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("😀 Emoji Emotion Classifier")
st.markdown("""
<center>
Predict whether an emoji is **Positive**, **Neutral**, or **Negative**.<br>
Type an emoji below and see the emotion instantly!
</center>
""", unsafe_allow_html=True)

st.write("---")

# Input box
emoji_input = st.text_input("Enter an emoji:", max_chars=2)

# Prediction display
if emoji_input:
    vec = vectorizer.transform([emoji_input])
    pred = model.predict(vec)
    emotion = le.inverse_transform(pred)[0]
    
    # Color-coded card with shadow effect
    if emotion == "Positive":
        st.markdown(f"""
        <div style="background-color:#d4edda; padding:20px; border-radius:15px; text-align:center; font-size:28px; color:#155724; box-shadow: 3px 3px 10px #888888;">
        Predicted Emotion: {emotion}
        </div>
        """, unsafe_allow_html=True)
    elif emotion == "Negative":
        st.markdown(f"""
        <div style="background-color:#f8d7da; padding:20px; border-radius:15px; text-align:center; font-size:28px; color:#721c24; box-shadow: 3px 3px 10px #888888;">
        Predicted Emotion: {emotion}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#e2e3e5; padding:20px; border-radius:15px; text-align:center; font-size:28px; color:#383d41; box-shadow: 3px 3px 10px #888888;">
        Predicted Emotion: {emotion}
        </div>
        """, unsafe_allow_html=True)

st.write("---")
st.markdown("<center>Powered by <b>Python, scikit-learn & Streamlit</b></center>", unsafe_allow_html=True)

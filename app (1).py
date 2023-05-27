import pathlib
from pathlib import Path
import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class Predict(): 
    def __init__(self, filename: str):
        self.learn_interface = load_learner(Path() / filename)
        self.img = self.get_image_from_upload()

        if self.img is not None:
            self.get_prediction()

    @staticmethod
    def get_image_from_upload():
        st.header("FrogApp")
        uploaded_file = st.file_uploader("Wybierz zabe do klasyfikacji", type= ['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            return PILImage.create(uploaded_file)

    def get_prediction(self):
        if st.button("Co to za gatunek?"):
            pred, pred_idx, probs = self.learn_interface.predict(self.img)
            prob = probs[pred_idx].item() * 100
            left, right = st.columns(2)
            with left:
                st.metric("Przewidziana klasa", pred)
            with right: 
                st.metric("Prawdopodobienstwo", "{0:.0f}".format(prob))

if __name__ == '__main__': 
    file_name = 'model.pkl'

Predict(file_name)
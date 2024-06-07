# Core Pkgs
import streamlit as st
st.set_page_config(
    page_title="COVID_DEEPLEARNING_APP",
    page_icon="🚑",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)
import os
import time
# Viz Pkgs
import cv2
from PIL import Image, ImageEnhance
import numpy as np
# AI Pkgs
import tensorflow as tf

# Initialize session state for the image
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

def main():
    """Simple Tool for Covid-19 Detection from Chest X-Ray"""
    html_templ = """
    <div style="background-color:cyan;padding:10px;">
    <h1 style="color:black">Covid-19 Detection Tool</h1>
    </div>
    """
    st.markdown(html_templ, unsafe_allow_html=True)
    st.write("A simple proposal for Covid-19 Diagnosis from X-Ray of Chest Images powered by Deep Learning and Streamlit")

    st.sidebar.image("gemini_img.jpeg", use_column_width=True)

    image_file = st.sidebar.file_uploader("Upload an X-Ray Image (jpg, png or jpeg)", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        if st.session_state.enhanced_image is None or st.session_state.original_image is None:
            st.session_state.enhanced_image = our_image
            st.session_state.original_image = our_image

        if st.sidebar.button("Image Preview"):
            st.sidebar.image(st.session_state.enhanced_image, width=300)

        activities = ["Image Enhancement", "Diagnosis", "Disclaimer and Info"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        if choice == 'Image Enhancement':
            st.subheader("Image Enhancement")

            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Contrast", "Brightness"])

            if enhance_type == "Contrast":
                c_rate = st.slider("Contrast", 0.5, 5.0)
                enhancer = ImageEnhance.Contrast(st.session_state.enhanced_image)
                st.session_state.enhanced_image = enhancer.enhance(c_rate)
                st.image(st.session_state.enhanced_image, width=500, use_column_width=True)
            elif enhance_type == "Brightness":
                b_rate = st.slider("Brightness", 0.5, 5.0)
                enhancer = ImageEnhance.Brightness(st.session_state.enhanced_image)
                st.session_state.enhanced_image = enhancer.enhance(b_rate)
                st.image(st.session_state.enhanced_image, width=500, use_column_width=True)
            else:
                st.session_state.enhanced_image = st.session_state.original_image
                st.text("Original Image")
                st.image(st.session_state.enhanced_image, width=500, use_column_width=True)

            # Add a button to revert to the original image
            if st.sidebar.button("Revert to Original"):
                st.session_state.enhanced_image = st.session_state.original_image
                st.image(st.session_state.enhanced_image, width=500, use_column_width=True)

        elif choice == "Diagnosis":
            if st.sidebar.button("Diagnose"):
                # Converting image to gray-scale
                new_img = np.array(st.session_state.enhanced_image.convert('RGB'))
                new_img = cv2.cvtColor(new_img, 1)  # 0 is original, 1 is greyscale (should be corrected)
                Gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                st.info("Chest X-Ray")
                st.image(Gray, width=400, use_column_width=True)
                # Pre-processing the image
                img_size = (200, 200)
                img = cv2.equalizeHist(Gray)  # increases clarity of image
                img = cv2.resize(img, img_size)
                img = img / 255  # normalization
                x_ray = img.reshape(1,1,200,200,1)
                # Loading the model
                model = tf.keras.models.load_model("./models/Covid19_CNN_Classifier.h5")
                # Prediction (Classifying into 2 categories, Covid or Not Covid)
                diagnosis_prob = model.predict(x_ray)
                diagnosis = np.argmax(diagnosis_prob, axis=1)

                my_bar = st.sidebar.progress(0)
                for value in range(100):
                    time.sleep(0.05)
                    my_bar.progress(value + 1)
                # Output Result
                if diagnosis == 0:
                    st.sidebar.success("DIAGNOSIS: YAYY! COVID NOT DETECTED")
                else:
                    st.sidebar.error("DIAGNOSIS: COVID DETECTED")
                st.warning("This Web App is just a DEMO about Streamlit and Artificial Intelligence and there is no clinical value in its diagnosis!")

        else:
            st.subheader("Disclaimer and Info")
            st.subheader("Disclaimer")
            st.write("**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**")
            st.write("**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**")
            st.subheader("Info")
            st.write("This Tool gets inspiration from the following works:")
            st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)") 
            st.write("- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)") 
            st.write("- [Deep Learning per la Diagnosi del COVID-19](https://www.youtube.com/watch?v=dpa8TFg1H_U&t=114s)")
            st.write("We used 206 Posterior-Anterior (PA) X-Ray [images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv) of patients infected by Covid-19 and 206 Posterior-Anterior X-Ray [images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of healthy people to train a Convolutional Neural Network (made by about 5 million trainable parameters) in order to make a classification of pictures referring to infected and not-infected people.")
            st.write("Since dataset was quite small, some data augmentation techniques have been applied (rotation and brightness range). The result was quite good since we got 94.5% accuracy on the training set and 89.3% accuracy on the test set. Afterwards the model was tested using a new dataset of patients infected by pneumonia and in this case the performance was very good, only 2 cases in 206 were wrongly recognized. Last test was performed with 8 SARS X-Ray PA files, all these images have been classified as Covid-19.")
            st.write("Unfortunately in our test we got 5 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.")
            st.write("The model is suffering of some limitations:")
            st.write("- small dataset (a bigger dataset for sure will help in improving performance)")
            st.write("- images coming only from the PA position")
            st.write("- a fine tuning activity is strongly suggested")
            st.write("")
            st.write("Anybody has interest in this project can drop me an email and I'll be very happy to reply and help.")

    if st.sidebar.button("About the Author"):
        st.markdown("[abhirupbasu30@gmail.com](mailto:abhirupbasu30@gmail.com)")

if __name__ == '__main__':
    main()
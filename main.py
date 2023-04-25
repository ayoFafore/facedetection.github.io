# Importing Required Libraries
import streamlit as st
from utils import get_detection
import io
from PIL import Image
import numpy as np



# Opening all list of objects 
prefix = "object_detection_model/"
labelsPath = prefix + "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")


# Website title
st.set_page_config(
    page_title="Face and Object Detection", 
)

# MBZUAI Logo
st.image('./MBZUAI_Logo.jpeg')

# Web Page Title
st.title("Face and Object Detection")


object_name = None
all_objects = None


image = st.file_uploader("Please select an image", type=['jpg', 'jpeg', 'png'])  # Image Upload/Browse button with allowed file extensions
if image is not None: # Check if image is uploaded
    st.image(image, width=600)  # Show the uploaded image
face_radio = st.radio("Face Detection?", options=['Yes', 'No']) # Radio button for enabling/disabling Face Detection
object_radio = st.radio("Object Detection?", options=['Yes', 'No']) # Radio button for enabling/disabling Object Detection
if object_radio == 'Yes': # Check If object detection is enabled
    all_objects = st.radio("All Objects", options=['Yes', 'No']) # Radio button for detection of all objects or selected one
    if all_objects == "No": # For selected one
        object_name = st.selectbox("Select an Object", options=LABELS) # DropDown for selecting a specific object

# Above all are different input parameters 

detect_button = st.button("Detect") # Finally a button to detect based on input parameters
if detect_button: # Check If button pressed 
    with st.spinner("Detecting objects and Faces."): # Waiting spinner while AI models are detecting objects
        if image is not None:  # Check if image is already uploaded
            image_byte = image.getvalue() # Getting image values in Bytes
            image_array = np.asarray(Image.open(io.BytesIO(image_byte))) # Converting Bytes to Array
            out_image, faces, objects = get_detection(image_array, face_radio, object_radio, all_objects, object_name) # Calling detections function with all parameters
    
    st.image(out_image) # Showing output image
    st.text("Face Coordinates")
    st.json(faces) # Showing Face metedata (Face Coordinates)
    st.text("Objects Coordinates")
    st.json(objects) # Showing Object metadata (Objects name and coordinates)


# Footer Style  

def footer():
    myargs = [
        "Made with ❤️ by Aamena ",
           ]


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>@Made by Aamena AlShehyari</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

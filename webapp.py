import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import PIL
from pothole_detection import detect_from_image,detect_from_video
import socket 
from pymongo import MongoClient
import geocoder
from mail import send_email

cluster = MongoClient('#---MongoDB Address Here--------#')
db = cluster["potholes"]


hostname = socket.gethostname() 
IPAddr = socket.gethostbyname(hostname)  
os.makedirs('uploads',exist_ok=True)



page = st.sidebar.selectbox("Pages Menu",options=['Home','Using Image','Using Video'])

g = geocoder.ip('me')
print(g.latlng)


def register(location,higway_type,size,position):
    #collection = db["potholes_report"]
    # try:
    dic = {"location":location,"highway_type":higway_type,"size":size,"position":position}
    send_email(dic, 'dev.isaaca@gmail.com')
    #collection.insert_one(dic)
    st.info("Reported")
    # except:
    #     print("There was a Issue While Reporting Data to server")
    #     st.warning("There was a Issue While Reporting Data to server")

def get_pothole_info():
    st.sidebar.markdown("***")
    st.info(f"Getting Location of pothole using your IP Address {g.ip}")
    df = pd.DataFrame([g.latlng],columns=['lat', 'lon'])

    st.map(df)
    st.info(f"Location According to IP address {g.city},{g.state}")
    location = (g.city,g.state)
    highway_type = st.sidebar.selectbox(label="Select Road Type:",options=["National Highway","Local Road"])
    size = st.sidebar.selectbox(label="Approx. Size of Pothole",options=["Small Pothole","Medium Pothole","Large Pothole"])
    position = st.sidebar.selectbox(label="Position of Pothole",options=["Center","Sideways"])

    return location,highway_type,size,position

def load_image(image_file):
    img = Image.open(image_file)
    img.save("uploads/image.jpg")
    return img

def load_video(video_file):
    path_name = "uploads/video.mp4"
    with open(path_name,'wb') as f:
        f.write(video_file.read())

if page == 'Using Image':
    
    st.title("Pothole Detection Using Image")
    choice_upload = st.sidebar.selectbox("Select a Method",options=['Upload Image','Open Camera'])
    if choice_upload == 'Upload Image':
        
        image_file = st.file_uploader('Upload Image',accept_multiple_files=False,type=['png','jpg','JPG','jpeg'])
        if image_file is not None:
            a,b = st.columns(2)
            file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
            st.write(file_details)
            a.image(load_image(image_file))
            detect_from_image("uploads/image.jpg")
            b.image("results/image_result.jpg")

            location,highway_type,size,position = get_pothole_info()

            submit_report = st.sidebar.button("Submit Report")
            if submit_report:
                register(location,highway_type,size,position)

    if choice_upload == 'Open Camera':

        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            img = Image.open(img_file_buffer)
            img = img.save("uploads/image.jpg")
            detect_from_image("uploads/image.jpg")
            st.image("results/image_result.jpg")

            location,highway_type,size,position = get_pothole_info()

            submit_report = st.sidebar.button("Submit Report")
            if submit_report:
                register(location,highway_type,size,position)


elif page == 'Using Video':
    st.title("Pothole Detection Using Video")

    st.warning("Video Processing will Take lot of Computization power.It can crash your system,Use GPU for better performance.")
    video_file = st.file_uploader(label="Upload Video",accept_multiple_files=False,type=["mp4","mkv","avi"])

    if video_file is not None:
        load_video(video_file)
        
        detect_from_video("uploads/video.mp4")
        os.system('ffmpeg -i results/video_result.avi -vcodec libx264 results/processed.mp4 -y')
        st.snow()

        video_result = open("results/processed.mp4",'rb')
        video_bytes = video_result.read()
        st.video(video_bytes)

        location,highway_type,size,position = get_pothole_info()

        submit_report = st.sidebar.button("Submit Report")
        if submit_report:
            register(location,highway_type,size,position)


        


else:
    st.title('Pothole Detection')
    st.markdown("> Select any Choice from sidebar to procced")

    st.image("model_files/out.jpg")
    cont = """

    ## Detecting Potholes on road using YOLO Model
    
    *Features:*
    - Detects Pothole From Image
    - Detects Pothole Using Live Camera Feed
    - Detects Potholes From Uploaded Video
    - Report Potholes Related Data to MonogDB
    - Automatically Gets Location Information using IP Address.
    - Detects Other elements from Image or Video.
    
    """

    st.markdown(cont)

    

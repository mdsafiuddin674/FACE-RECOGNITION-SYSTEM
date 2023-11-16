import csv
import datetime
import time
import tkinter as tk

import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

window = tk.Tk()

window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'

window.configure(background='blue')
window.geometry("1600x800")
window.state('zoomed')

# window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Smart Face-Py Monitor", bg="Green", fg="white", width=50, height=3,
                   font=('times', 30, 'italic bold underline'))
message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="red", bg="yellow", font=('times', 15, ' bold '))
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="red", bg="yellow", height=2, font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ", width=20, fg="red", bg="yellow", height=2,
                font=('times', 15, ' bold underline '))
lbl3.place(x=400, y=400)

message = tk.Label(window, text="", bg="yellow", fg="red", width=30, height=2, activebackground="yellow",
                   font=('times', 15, ' bold '))
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ", width=20, fg="red", bg="yellow", height=2,
                font=('times', 15, ' bold  underline'))
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="", fg="red", bg="yellow", activeforeground="green", width=30, height=2,
                    font=('times', 15, ' bold '))
message2.place(x=700, y=650)


def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def take_images():
    user_id = (txt.get())
    name = (txt2.get())
    if is_number(user_id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        har_cascade_path = "har_cascade_frontal_face_default.xml"
        detector = cv2.CascadeClassifier(har_cascade_path)
        sample_num = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sample_num = sample_num + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite('TrainingImage\\' + name + "." + user_id + '.' + str(sample_num) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 milliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is more than 100
            elif sample_num > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + user_id + " Name : " + name
        row = [user_id, name]
        with open('StudentDetails\\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if is_number(user_id):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if name.isalpha():
            res = "Enter Numeric Id"
            message.configure(text=res)


def train_images():
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    har_cascade_path = "har_cascade_frontal_face_default.xml"
    detector = cv2.CascadeClassifier(har_cascade_path)
    faces, user_id = get_images_and_labels("TrainingImage")
    recognizer.train(faces, np.array(user_id))
    recognizer.save("TrainingImageLabel\\Trainner.yml")
    res = "Image Trained"  # +",".join(str(f) for f in user_id)
    message.configure(text=res)


def get_images_and_labels(path):
    # get the path of all the files in the folder
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(image_paths)

    # create empty face list
    faces = []
    # create empty ID list
    ids = []
    # now looping through all the image paths and loading the ids and the images
    for image_path in image_paths:
        # loading the image and converting it to gray scale
        pill_image = Image.open(image_path).convert('L')
        # Now we are converting the PIL image into numpy array
        image_np = np.array(pill_image, 'uint8')
        # getting the user_id from the image
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(image_np)
        ids.append(user_id)
    return faces, ids


def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\\Trainner.yml")
    har_cascade_path = "har_cascade_frontal_face_default.xml"
    face_cascade = cv2.CascadeClassifier(har_cascade_path)
    df = pd.read_csv("StudentDetails\\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            user_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == user_id]['Name'].values
                tt = str(user_id) + "-" + aa
                attendance.loc[len(attendance)] = [user_id, aa, date, time_stamp]

            else:
                user_id = 'Unknown'
                tt = str(user_id)
            if conf > 75:
                no_of_file = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\\Image" + str(no_of_file) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if cv2.waitKey(1) == ord('q'):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    hour, minute, second = time_stamp.split(":")
    file_name = "Attendance\\Attendance_" + date + "_" + hour + "-" + minute + "-" + second + ".csv"
    # if date.getTimeInHour() >= 9 and date.getTimeInHour() <= 11:
    #     file_name = "Attendance\Attendance_Login" + date + ".csv"
    # elif date.getTimeInHour() >= 16 and date.getTimeInHour() <= 18:
    #     file_name = "Attendance\Attendance_Logout" + date + ".csv"

    attendance.to_csv(file_name, index=False)
    cam.release()
    cv2.destroyAllWindows()
    # print(attendance)
    res = attendance
    message2.configure(text=res)


clearButton = tk.Button(window, text="Clear", command=clear, fg="red", bg="yellow", width=20, height=2,
                        activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="red", bg="yellow", width=20, height=2,
                         activebackground="Red", font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Take Images", command=take_images, fg="red", bg="yellow", width=20, height=3,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=train_images, fg="red", bg="yellow", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=track_images, fg="red", bg="yellow", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="red", bg="yellow", width=20, height=3,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,
                    font=('monospace', 18, 'bold'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Mohd Danish, Mohammad Tabish & Md Safiuddin")
copyWrite.configure(state="disabled", fg="grey")
copyWrite.pack(side="left")
copyWrite.place(x=790, y=760)

window.mainloop()

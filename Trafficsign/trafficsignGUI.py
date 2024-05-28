import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('model.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Class 1: Speed limit (20km/h)',
            2:'Class 2: Speed limit (30km/h)',      
            3:'Class 3: Speed limit (50km/h)',       
            4:'Class 4: Speed limit (60km/h)',      
            5:'Class 5: Speed limit (70km/h)',    
            6:'Class 6: Speed limit (80km/h)',      
            7:'Class 7: End of speed limit (80km/h)',     
            8:'Class 8: Speed limit (100km/h)',    
            9:'Class 9: Speed limit (120km/h)',     
           10:'Class 10: No passing',   
           11:'Class 11: No passing veh over 3.5 tons',     
           12:'Class 12: Right-of-way at intersection',     
           13:'Class 13: Priority road',    
           14:'Class 14: Yield',     
           15:'Class 15: Stop',       
           16:'Class 16: No vehicles',       
           17:'Class 17: Veh > 3.5 tons prohibited',       
           18:'Class 18: No entry',       
           19:'Class 19: General caution',     
           20:'Class 20: Dangerous curve left',      
           21:'Class 21: Dangerous curve right',   
           22:'Class 22: Double curve',      
           23:'Class 23: Bumpy road',     
           24:'Class 24: Slippery road',       
           25:'Class 25: Road narrows on the right',  
           26:'Class 26: Road work',    
           27:'Class 27: Traffic signals',      
           28:'Class 28: Pedestrians',     
           29:'Class 29: Children crossing',     
           30:'Class 30: Bicycles crossing',       
           31:'Class 31: Beware of ice/snow',
           32:'Class 32: Wild animals crossing',      
           33:'Class 33: End speed + passing limits',      
           34:'Class 34: Turn right ahead',     
           35:'Class 35: Turn left ahead',       
           36:'Class 36: Ahead only',      
           37:'Class 37: Go straight or right',      
           38:'Class 38: Go straight or left',      
           39:'Class 39: Keep right',     
           40:'Class 40: Keep left',      
           41:'Class 41: Roundabout mandatory',     
           42:'Class 42: End of no passing',      
           43:'Class 43: End no passing veh > 3.5 tons' }
                 
#initialise GUI
top=tk.Tk()
top.geometry('600x600')
top.title('Traffic sign classification project ')
top.configure(background='#ffffff')

label=Label(top,background='#ffffff', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    # print(image.shape)
# predict classes
    pred_probabilities = model.predict(image)[0]
    print(pred_probabilities)
    pred = pred_probabilities.argmax(axis=-1)
    sign = classes[pred+1]
    # print(sign)
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify ",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#c71b20', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#c71b20', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Traffic sign classification project",pady=10, font=('arial',20,'bold'))
heading.configure(background='#ffffff',foreground='#364156')


heading.pack()
top.mainloop()
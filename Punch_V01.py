import random
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.cache import Cache
from kivy.clock import Clock

import win32timezone
import threading
from scipy.fft import *
from scipy import signal
from scipy.fftpack import *
import pyaudio
import os
import cv2
import numpy as np 

audio_buffer = np.array([])
punch_value_lock = threading.Lock()
eyelx,eyely,eyerx,eyery = 0,0,0,0
can_call_left_punch = True

class MyApp(App):
    def build(self):
        #main box
        self.layout = BoxLayout(orientation='vertical')
        self.count = 0
        self.samplerate = 44100
        self.image_path = ""
        self.Ori_Pic_source = ""
        self.audio_thread = None
        self.audio_thread_stop_flag = threading.Event()
     
        self.open_button = Button(text='Select Pic', size_hint=(.1, .1), pos_hint={'x':.0, 'y':.5}, disabled=False)
        self.open_button.bind(on_press=self.open_file_chooser)
        
        # audio thread button
        self.start_audio_button = Button(text='Start Game', size_hint=(.1, .1), pos_hint={'x':.0, 'y':.5}, disabled=True)
        self.start_audio_button.bind(on_press=self.start_audio_thread)

        self.reset_button = Button(text='Restart', size_hint=(.1, .1), pos_hint={'x':.0, 'y':.5}, disabled=True)
        self.reset_button.bind(on_press=self.Clear_plot)
        # self.punch_button = Button(text='punch', disabled=True)
        # self.punch_button.bind(on_press=self.handle_punch_wrapper)
        
        # Image widget
        self.image = Image()

        # widget
        self.layout.add_widget(self.open_button)
        self.layout.add_widget(self.start_audio_button)
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.reset_button)
        # self.layout.add_widget(self.punch_button)
        return self.layout

    def open_file_chooser(self, instance):
        filechooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp'])
        filechooser.path = os.getcwd()

        filechooser.bind(on_submit=self.on_file_select)
        self.layout.add_widget(filechooser)
        self.open_button.disabled = True
        self.start_audio_button.disabled = True
        
    def Clear_plot(self, instance):
        print("clear plot")
        if self.audio_thread:
            self.audio_thread_stop_flag.set()
            self.audio_thread.join()  
            self.audio_thread = None  

        self.image.source = self.Ori_Pic_source
        self.count = 0
        self.open_button.disabled = False
        self.start_audio_button.disabled = False
        self.reset_button.disabled = True
        
    def detect_face(self,image_path):
        print("face detect")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # eyelx,eyely,eyerx,eyery = self.find_eye(image_path)
        
        # 偵測臉部
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        

        # 檢查是否偵測到臉部
        return len(faces) > 0
    
    def update_image(self, new_image_path):
        print("update pic")
        self.image.source = new_image_path
        self.count += 1
    
    def handle_punch_wrapper(self, instance):
        print("Call Punch")
        global audio_buffer, can_call_left_punch
        if self.image.source:
            self.dirc = instance
            self.handle_punch(self.image.source, self.dirc, self.count)


    def on_file_select(self, filechooser, selection, touch):
        print("Select Done")
        global eyelx,eyely,eyerx,eyery

        if selection:

            if self.detect_face(selection[0]):
                self.image.source = selection[0]
                self.Ori_Pic_source = selection[0]
                print(selection[0])
                eyelx,eyely,eyerx,eyery = self.find_eye(self.image.source)
                
                self.layout.remove_widget(filechooser)
                self.start_audio_button.disabled = False
                # self.punch_button.disabled = False
                self.open_button.disabled = True
            else:
                
                popup = Popup(title='Tip',
                            content=Label(text='Not human face, Please select again'),
                            size_hint=(None, None), size=(400, 400))
                popup.open()
                
    def start_audio_thread(self, instance):
        print("Start Game")
        self.audio_thread_stop_flag.clear()
        self.audio_thread = threading.Thread(target=self.audio_processing)  
        self.audio_thread.daemon = True
        self.audio_thread.start()
        self.start_audio_button.disabled = True
        self.open_button.disabled = True
        self.reset_button.disabled = False
        
    def convex(self, src_img, raw, effect):
        print("fish eyes effect")
        col, row, channel = raw[:]     
        cx, cy, r = effect[:]           
        output = np.zeros([row, col, channel], dtype = np.uint8)        
        for y in range(row):
            for x in range(col):
                d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) ** 0.5  
                if d <= r:
                    nx = int((x - cx) * d / r + cx)        
                    ny = int((y - cy) * d / r + cy)        
                    output[y, x, :] = src_img[ny, nx, :]   
                else:
                    output[y, x, :] = src_img[y, x, :]     
        return output
    
    def paste_png(self, img, fu, harm, dirc, eyelx,eyely,eyerx,eyery): # dirc: left or right
        print("paste process")
        c1x,c1y,c2x,c2y = eyelx,eyely,eyerx,eyery
        ma_x1_width = 100
        ma_y1_width = 100

        ran_x1_width, ran_y1_width =  random.randint(int(ma_x1_width/2),int(ma_x1_width)), \
        random.randint(int(ma_y1_width/2),int(ma_y1_width))
        ba = img
        if dirc == 'l':

            c1x_start, c1y_start = int(np.mean(c1x)), int(np.mean(c1y))
        else:
            c1x_start, c1y_start = int(np.mean(c2x)), int(np.mean(c2y))
        
        shift = 30
        c1x_start -= shift
        background = ba[int(c1y_start):int(c1y_start+ran_y1_width),
                int(c1x_start):int(c1x_start+ran_x1_width)]

        overlay = cv2.imread(fu, cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
        dim = (background.shape[1],background.shape[0])
        overlay = cv2.resize(overlay,dim , interpolation = cv2.INTER_AREA)
        height, width = overlay.shape[:2]
        print(background.shape,overlay.shape)
        for y in range(height):
            for x in range(width):
                overlay_color = overlay[y, x, :3]  # first three elements are color (RGB)
                overlay_alpha = overlay[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0

        # get the color from the background image
                background_color = background[y, x]

        # combine the background color and the overlay color weighted by alpha
                composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha

        # update the background image in place
                background[y, x] = composite_color
        ba[int(c1y_start):int(c1y_start+ran_y1_width),int(c1x_start):int(c1x_start+ran_x1_width)]
        #for i in range(15):
        ranx = random.randint(int(c1x_start),int(c1x_start+ran_x1_width))
        rany = random.randint(int(c1y_start),int(c1y_start+ran_y1_width))
        #test = convex(ba, (ba.shape[1], ba.shape[0], 3), 
        #(int((c1y_start+ran_y1_height//2)), int((c1x_start+ran_x1_width//2)),  (i+1)*5))
        ba = self.convex(ba, (ba.shape[1], ba.shape[0], 3), 
        (ranx, rany,  harm*5))
        return ba
            
    def find_eye(self,img):
        print("check face process")
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        LBFmodel = "lbfmodel.yaml"
        faces = haarcascade.detectMultiScale(gray,1.3,5)
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodel)

    #print(faces)
        _, landmarks = landmark_detector.fit(gray, faces)
    #eye = []
        reye = landmarks[0][0][36:41]
        leye = landmarks[0][0][42:48]
        reye_center_x, reye_center_y,leye_center_x, leye_center_y = 0, 0, 0, 0
        for i in range(len(reye)):
            reye_center_x += reye[i][0]
            reye_center_y += reye[i][1]
        reye_center_x =  int(reye_center_x // len(reye))
        reye_center_y =  int(reye_center_y // len(reye))

        for i in range(len(leye)):
            leye_center_x += leye[i][0]
            leye_center_y += leye[i][1]
        leye_center_x =  int(leye_center_x // len(leye))
        leye_center_y =  int(leye_center_y // len(leye))
        # print(reye_center_x,reye_center_y,leye_center_x,leye_center_y)

        return reye_center_x,reye_center_y,leye_center_x,leye_center_y     
       
    def handle_punch(self, dirc):
        global eyelx,eyely,eyerx,eyery
        print("handle process")
        path = os.path.dirname(self.image.source)
        directory_with_slash = path + "\\"
        file_name = os.path.basename(self.image.source)
        # print(f"path{directory_with_slash}")
        new_img = directory_with_slash+'temp'+str(self.count)+'.jpg'
      
        # print(new_img)
        print(self.count ,"_@@@@@@@@@@@@@@")
        dirc = dirc
        harm = 5
        img = cv2.imread(self.image.source)

        if self.count<=0:
            img = cv2.imread(self.image.source)
        else:
            #print("#################",path+ new_img )
            next_img = directory_with_slash+'temp'+str(self.count-1)+'.jpg'
            img = cv2.imread(next_img)

        # print(selection)
        fu_path  = directory_with_slash+"fu1.png"
        ret = self.paste_png(img, fu_path, harm, dirc, eyelx, eyely, eyerx, eyery)
        print("saving pic..:", new_img)
        success = cv2.imwrite(new_img, ret)
        if not success:
            print(f"saving pic fail {new_img}")
        else:
            print(f"saving pic success {new_img}")
        # print(new_img)
        cv2.imwrite(new_img, ret)
        # cv2.imshow(new_img, ret)
        # self.image.source = ''
        
        # self.update_image(new_img)
        Clock.schedule_once(lambda dt: self.update_image(new_img))

        print("handle_done")
        
        
    def FR_shift(self,data,l):
        max_value = np.max(data)
        x = np.linspace(1, len(data), len(data))
        area = np.trapz(data, x)
        return float("{:.2f}".format(round(area/max_value, 2))), float("{:.2f}".format(round(area, 2)))

    # Punch Information Analysis
    def punch_info(self,data): 
        # print(data) # OK
        sr = 44100
        ind_18k, ind_20k = 105, 116
        sig = hilbert(data)
        amp_envelope = np.abs(sig)
        phase = np.unwrap(np.angle(sig))
        f, t, Zxx = signal.stft(data, sr)
        Zxx = np.abs(Zxx)
        
        shift = 4
        data18_f, data20_f, data18_zxx, data20_zxx = \
        f[ind_18k-shift:ind_18k+shift], f[ind_20k-shift:ind_20k+shift], Zxx[ind_18k-shift:ind_18k+shift,:],\
        Zxx[ind_20k-shift:ind_20k+shift,:]
        
        yf = fft(data)
        yf = np.abs(yf)
        N = len(data)
        dt = 2/sr/N
        xf = fftfreq(N, dt)
        yf = yf[2800:4800]
        yf = yf[0:750]
        fre18_s, fre18_e, fre20_s, fre20_e = 200, 350, 500, 650 
        # ma18, ma20 = max(yf[fre18_s:fre18_e]), max(yf[fre20_s:fre20_e])

        try:
            ma18, ma20 = max(yf[fre18_s:fre18_e]), max(yf[fre20_s:fre20_e])
            shift18, shift20 = self.FR_shift(yf[fre18_s:fre18_e], 0), self.FR_shift(yf[fre20_s:fre20_e], 0)
            cut_max, cut_area = 20, 10
            
            if ma18 > cut_max and shift18[0] > cut_area and shift18[1] > shift20[1]:
                return 'l'
            if ma20 > cut_max and shift20[0] > cut_area and shift20[1] > shift18[1]:
                return 'r'

        except ValueError:
            # Handle the error, maybe log it or set default values
            ma18, ma20 = 0, 0
            print("Encountered empty sequence in frequency analysis.")

        return False

    def enable_left_punch(self):
        global can_call_left_punch
        can_call_left_punch = True

    def audio_processing(self):

        p = pyaudio.PyAudio()


        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.samplerate,
                        input=True,
                        frames_per_buffer=1024)

        try:
            while not self.audio_thread_stop_flag.is_set():

                data = stream.read(1024)
 
                self.audio_callback(data)

        finally:

            stream.stop_stream()
            stream.close()

            p.terminate()


    def audio_callback(self, indata):
        global audio_buffer,can_call_left_punch

        audio_data = np.frombuffer(indata, dtype=np.int16)      
        float_data = audio_data / 32768.0
        audio_buffer = np.concatenate((audio_buffer, float_data))
        # print(audio_data)
        # print(audio_buffer)
        while len(audio_buffer) >= 7500:
            data_to_process = audio_buffer[:7500]
            audio_buffer = audio_buffer[7500:]

            # try:
            punch_result = self.punch_info(data_to_process)
            # print(punch_result)
            
            if punch_result == 'l' or punch_result == 'r':
                if can_call_left_punch:
                    with punch_value_lock:
                        # punch_value = punch_result
                        # print(punch_value)
                        
                        # Clock.schedule_once(lambda dt: self.handle_punch_wrapper(punch_result))
                        self.handle_punch(punch_result)
                        can_call_left_punch = False
                        
                        audio_buffer = np.array([])
                        threading.Timer(0.8, self.enable_left_punch).start()
                        print("detect punch")
                        
                        
            # except Exception as e:
            #     # punch_value = None
            #     print(f"An error occurred: {e}")


if __name__ == '__main__':
    MyApp().run()

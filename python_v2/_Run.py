import sys
sys.path.append('../Library')



import argparse
import pygame
import threading
import time
import multiprocessing
# import SpoutSDK

from queue import Queue, LifoQueue
from multiprocessing import Pipe, Process

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

from realtime_demo import *
from yolo import *

#-----------------------------

DETECT_FACE=True
DETECT_EMOTION=True
DETECT_OBJ=True

USE_THREAD=True

USE_SPOUT=False


#-----------------------------
# osc initialization
IP="127.0.0.1"
PORT=5555
client=udp_client.SimpleUDPClient(IP, PORT)
#-----------------------------


WIDTH=768
HEIGHT=432
SPOUT_NAME="Camera"


#-----------------------------

dirname = os.path.dirname(__file__)
yolo_config={
	"weights_path": os.path.join(dirname, '../model/yolo3_tiny/yolov3-tiny.h5'),
    "anchors_path": os.path.join(dirname, '../configs/tiny_yolo3_anchors.txt'),
    "classes_path": os.path.join(dirname, '../configs/coco_classes.txt'),
}

#-----------------------------

face_cascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')


#-----------------------------

print_lock = threading.Lock()
result_lock=threading.Lock()



class DetectFace(threading.Thread):
	def __init__(self,queue):
		threading.Thread.__init__(self)
		print("Init Face Detect...")
		self.face_detect = FaceCV(depth=16, width=8)
		self.result=LifoQueue()
		self.queue=queue

	def run(self):
		while True:
			input_faces=self.queue.get()
			if input_faces is None:
				continue

			faces, ages, genders=self.face_detect.detect_face_frame(input_faces)

			result=[]
			for i, face in enumerate(faces):
				gender = "F" if genders[i][0]>0.5 else "M"
				label = f"{int(ages[i])},{gender}"
				# with print_lock:
				# 	print(f"get face : {ages[i]} {gender}")

				data=[int(face[0]),int(face[1]),int(face[2]),int(face[3]),gender, ages[i],i]
				result.append(data)
			
			self.result.put(result)

class DetectEmotion(threading.Thread):
	def __init__(self,queue):
		threading.Thread.__init__(self)
		print("Init Emotion Detect...")
		self.emotion_model = model_from_json(open("../model/facial_expression_model_structure.json", "r").read())
		self.emotion_model.load_weights('../model/facial_expression_model_weights.h5') #load weights
		self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
		# self.face_cascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')
		self.queue=queue
		self.result=LifoQueue()

	def run(self):
		while True:
			(frame, faces)=self.queue.get()
			if frame is None:
				continue
			# print(f"emotion to detect= {len(faces)}")
			result=[]
			for i, (x,y,w,h) in enumerate(faces):
				
				detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
				detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
				detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				
				img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
				
				emotion_predictions = self.emotion_model.predict(img_pixels) #store probabilities of 7 expressions
				
				#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
				max_index = np.argmax(emotion_predictions[0])
				emotion = self.emotions[max_index]
				data=[int(x), int(y),int(w), int(h), emotion, max_index,i]
				result.append(data)

			self.result.put(result)
			# print(f"in thread: {self.result}")

class DetectObj(threading.Thread):
	def __init__(self, queue):
		threading.Thread.__init__(self)
		print("Start YOLO Thread...")
		self.yolo = YOLO_np(yolo_config)
		self.queue=queue
		self.output=LifoQueue()
		self.result=[]

	def run(self):
		while True:
			
			frame=self.queue.get()
			if frame is None:
				continue

			# with print_lock:
			with result_lock:
				# self.result.clear()
				result=[]
				image = Image.fromarray(frame)
				boxes, classes, scores=self.yolo.detect_image(image)

				class_count={}
				# print(f"new obj detect frame! #{len(classes)}")
				for i in range(len(classes)):
					ind=classes[i]
					if class_count.get(ind)==None:
						class_count[ind]=1
					else:
						class_count[ind]=class_count[ind]+1

				for i, box in enumerate(boxes):
					# label = f"{classes[i]}, {scores[i]}"
					data=[classes[i], int(box[0]), int(box[1]),int(box[2]-box[0]), int(box[3]-box[1]), float(scores[i]), int(i), int(class_count[classes[i]])]
					result.append(data)
				# print(f"detect= {len(self.result)}")
			self.output.put(result)



def draw_predict(img, point, size, label, color):
	# print(f"{point} {size}")
	x, y = point
	w, h= size
	cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
	cv2.putText(img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():

	global DETECT_FACE, DETECT_EMOTION, DETECT_OBJ, USE_THREAD, USE_SPOUT

	if not USE_SPOUT:
		video_capture = cv2.VideoCapture(0)
	else:
		display = (WIDTH,HEIGHT)
		pygame.init() 
		pygame.display.set_caption('Spout Receiver')
		pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
		# OpenGL init
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0,WIDTH,HEIGHT,0,1,-1)
		glMatrixMode(GL_MODELVIEW)
		glDisable(GL_DEPTH_TEST)
		glClearColor(0.0,0.0,0.0,0.0)
		glEnable(GL_TEXTURE_2D)


		#init spout
		spoutReceiver = SpoutSDK.SpoutReceiver()
		spoutReceiver.pyCreateReceiver(SPOUT_NAME,WIDTH,HEIGHT, False)
		textureReceiveID = glGenTextures(1)    


		# initalise receiver texture
		glBindTexture(GL_TEXTURE_2D, textureReceiveID)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

		# copy data into texture
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, None ) 
		glBindTexture(GL_TEXTURE_2D, 0)


	# if DETECT_FACE:
	# 	face_detect = FaceCV(depth=16, width=8)

	# if DETECT_OBJ:
	# 	yolo = YOLO_np(yolo_config)

	
	if DETECT_FACE:
		face_queue=Queue()
		detectFace=DetectFace(face_queue)
		detectFace.start()

	if DETECT_EMOTION:
		emotion_queue=Queue()
		detectEmotion=DetectEmotion(emotion_queue)
		detectEmotion.start()

	if DETECT_OBJ:
		obj_queue=Queue()
		obj_result=Queue()
		detectObj=DetectObj(obj_queue)
		detectObj.start()

	accum_time = 0
	curr_fps = 0
	fps = "FPS: ??"
	prev_time = timer()
	while True:	

		if USE_SPOUT:
			spoutReceiver.pyReceiveTexture(SPOUT_NAME, int(WIDTH), int(HEIGHT), int(textureReceiveID), GL_TEXTURE_2D, False, 0)
			glBindTexture(GL_TEXTURE_2D, textureReceiveID)
			# copy pixel byte array from received texture - this example doesn't use it, but may be useful for those who do want pixel info      
			frame = glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, outputType=None)  #Using GL_RGB can use GL_RGBA 

			# swap width and height data around due to oddness with glGetTextImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
			frame.shape = (frame.shape[1], frame.shape[0], frame.shape[2])
		else:
			ret, frame = video_capture.read()
		
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)


		if DETECT_FACE:
			face_queue.put((frame,faces))
			# faces, ages, genders=face_detect.detect_face_frame(frame)

			# for i, face in enumerate(faces):
			# 	gender = "F" if genders[i][0]>0.5 else "M"
			# 	label = f"{int(ages[i])},{gender}"
			# 	draw_predict(frame, (face[0], face[1]), (face[2], face[3]), label,(255,0,0))
			# 	client.send_message("/face", [int(face[0]),int(face[1]),int(face[2]),int(face[3]),gender, ages[i]])
			# print(f"Find {len(faces)} faces")
		
		if DETECT_EMOTION:
			emotion_queue.put((frame,faces))
			

		if DETECT_OBJ:
			obj_queue.put(frame)
			# image = Image.fromarray(frame)
			# boxes, classes, scores=yolo.detect_image(image)

			# class_count={}
			# for i in range(classes):
			# 	ind=int(classes[i])
			# 	if class_count.get(ind)==None:
			# 		class_count[ind]=1
			# 	else:
			# 		class_count[ind]=class_count[ind]+1

			# for i, box in enumerate(boxes):
			# 	label = f"{classes[i]}, {scores[i]}"
			# 	draw_predict(frame, (box[0], box[1]),(box[2]-box[0],box[3]-box[1]), label, (0,0,255))
				
			# 	data=[classes[i], int(box[0]), int(box[2]),int(box[1]), int(box[3]), float(score[i]), int(i), int(class_count[i])]
			# 	client.send_message("/detect", [classes[i], age[i]])
			# print(f"#obj={len(boxes)}")

		if DETECT_EMOTION:
			tmp_emotion=detectEmotion.result.get()

		if DETECT_FACE:
			# print(detectFace.result.qsize())
			tmp_face=detectFace.result.get()
			for i, result in enumerate(tmp_face):
				label=f"id= {result[6]}: {result[4]}, {result[5]}"
			
				#combine emotion
				if DETECT_EMOTION:
					label+=f", {tmp_emotion[i][4]}, {tmp_emotion[i][5]}"

				draw_predict(frame, (result[0], result[1]),(result[2],result[3]), label, (0,255,0))


		if DETECT_OBJ:
			# with result_lock:
			tmp=detectObj.output.get()
			if tmp:
			# print(f"obj result= {len(detectObj.result)}")
				for i, result in enumerate(tmp):
					# print(result)
					label=f"{result[0]},{result[5]} {result[6]}//{result[7]}"
					draw_predict(frame, (result[1], result[2]),(result[3],result[4]), label, (255,255,0))
			# detectObj.result.clear()
			

		# fps
		curr_time = timer()
		exec_time = curr_time - prev_time
		prev_time = curr_time
		curr_fps = 1/ exec_time
		cv2.putText(frame, text=f"fps= {curr_fps}", org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 255, 255), thickness=2)

		cv2.imshow('_Run', frame)

		key=cv2.waitKey(1)

		if key == 27:  # ESC key press
			break
		elif key==ord('a'):
			DETECT_FACE=not DETECT_FACE
			
		elif key==ord('s'):
			DETECT_EMOTION=not DETECT_EMOTION

		elif key==ord('d'):
			DETECT_OBJ=not DETECT_OBJ
			
		elif key==ord('t'):
			USE_THREAD=not USE_THREAD

	
	if not USE_SPOUT:
		video_capture.release()
	else:
		spoutReceiver.ReleaseReceiver()
	cv2.destroyAllWindows()
	
	detectEmotion.join()
	detectFace.join()
	detectObj.join()

	

if __name__ == "__main__":
	main()
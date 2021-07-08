import sys
sys.path.append('../Library')

import argparse
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder

from realtime_demo import *
from yolo import *



DETECT_FACE=True
DETECT_OBJ=False
USE_SPOUT=True

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


def draw_predict(img, point, size, label, color):
	# print(f"{point} {size}")
	x, y = point
	w, h= size
	cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
	cv2.putText(img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():

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


	if DETECT_FACE:
		face_detect = FaceCV(depth=16, width=8)

	if DETECT_OBJ:
		yolo = YOLO_np(yolo_config)

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
		
		if DETECT_FACE:
			faces, ages, genders=face_detect.detect_face_frame(frame)

			for i, face in enumerate(faces):
				gender = "F" if genders[i][0]>0.5 else "M"
				label = f"{int(ages[i])},{gender}"
				draw_predict(frame, (face[0], face[1]), (face[2], face[3]), label,(255,0,0))
				client.send_message("/face", [int(face[0]),int(face[1]),int(face[2]),int(face[3]),gender, ages[i]])
			# print(f"Find {len(faces)} faces")
		
		if DETECT_OBJ:
			image = Image.fromarray(frame)
			boxes, classes, scores=yolo.detect_image(image)

			class_count={}
			for i in range(classes):
				ind=int(classes[i])
				if class_count.get(ind)==None:
					class_count[ind]=1
				else:
					class_count[ind]=class_count[ind]+1

			for i, box in enumerate(boxes):
				label = f"{classes[i]}, {scores[i]}"
				draw_predict(frame, (box[0], box[1]),(box[2]-box[0],box[3]-box[1]), label, (0,0,255))
				
				data=[classes[i], int(box[0]), int(box[2]),int(box[1]), int(box[3]), float(score[i]), int(i), int(class_count[i])]
				client.send_message("/detect", [classes[i], age[i]])
			# print(f"#obj={len(boxes)}")


		# fps
		curr_time = timer()
		exec_time = curr_time - prev_time
		prev_time = curr_time
		curr_fps = 1/ exec_time
		cv2.putText(frame, text=f"fps= {curr_fps}", org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)

		cv2.imshow('_Run', frame)

		if cv2.waitKey(5) == 27:  # ESC key press
			break
	
	if not USE_SPOUT:
		video_capture.release()
	else:
		spoutReceiver.ReleaseReceiver()
	cv2.destroyAllWindows()
	

if __name__ == "__main__":
	main()
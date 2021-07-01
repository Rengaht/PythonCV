import numpy as np
import cv2
# from tensorflow import keras
from tensorflow.keras.preprocessing import image
from __GenderTest import *
import time


#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#-----------------------------
#face expression recognizer initialization
from tensorflow.keras.models import model_from_json

emotion_model = model_from_json(open("../model/facial_expression_model_structure.json", "r").read())
emotion_model.load_weights('../model/facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

age_model=ageModel()
gender_model=genderModel()

output_indexes = np.array([i for i in range(0, 101)])

prev_frame_time=0


while(True):
	ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#print(faces) #locations of detected faces

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face1 = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face1)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		emotion_predictions = emotion_model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(emotion_predictions[0])
		emotion = emotions[max_index]


		try:
			#age gender data set has 40% margin around the face. expand detected face.
			margin = 30
			margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
			detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
		except:
			print("detected face has no margin")
		
		try:
			detected_face2 = cv2.resize(detected_face, (224, 224))
		
			img_pixels2 = image.img_to_array(detected_face2)
			img_pixels2 = np.expand_dims(img_pixels2, axis = 0)
			img_pixels2 /= 255

			age_distributions = age_model.predict(img_pixels2)
			apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
					
			gender_distribution = gender_model.predict(img_pixels2)[0]
			gender_index = np.argmax(gender_distribution)
					
			if gender_index == 0: gender = "F"
			else: gender = "M"

		except Excetion as e:
			print("excetion", str(e))


		#fps
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		
		#write emotion text above rectangle
		output_attr=emotion+" / "+gender+" / "+apparent_age

		cv2.putText(img, output_attr, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
		cv2.putText(img, str(fps), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		#process on detected face end
		#-------------------------

	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()
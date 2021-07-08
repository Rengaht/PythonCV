from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder

from realtime_demo import *
from yolo import *


DETECT_FACE=True
DETECT_OBJ=False

#-----------------------------
# osc initialization
IP="127.0.0.1"
PORT=5555
client=udp_client.SimpleUDPClient(IP, PORT)
#-----------------------------


dirname = os.path.dirname(__file__)
yolo_config={
	"weights_path": os.path.join(dirname, '../model/yolo3_tiny/yolov3-tiny.h5'),
    "anchors_path": os.path.join(dirname, '../configs/tiny_yolo3_anchors.txt'),
    "classes_path": os.path.join(dirname, '../configs/coco_classes.txt'),
}

def draw_predict(img, point, size, label, color):
	# print(f"{point} {size}")
	x, y = point
	w, h= size
	cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
	cv2.putText(img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():
	video_capture = cv2.VideoCapture(0)

	if DETECT_FACE:
		face_detect = FaceCV(depth=16, width=8)

	if DETECT_OBJ:
		yolo = YOLO_np(yolo_config)

	accum_time = 0
	curr_fps = 0
	fps = "FPS: ??"
	prev_time = timer()
	while True:	
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
	
	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
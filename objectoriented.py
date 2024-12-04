# https://docs.opncv.ord/4.x/dd/d43/tutorial_py_video_display.html

import cv2
import numpy as np

print('hello world')
#this will be an ojected oriented  of the game
print('starting O . o . O Virtual3D')

class FaceFinder:
		"""Initializes a facecascade haar cascade filter to detect face in frame"""
		def __init__(self, ):
			print('FaceFinder O . o . O Intialize')
			self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		def find_face(self, frame):
			"""Returns face center (X,Y) draws Rect on a frame"""
			# Convert to Grayscale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = self.face_cascade.detectMultiScale(gray, minNeighbors = 9)
			print('detected Face(s) at:', faces)

			#Draw Rrectangle
			if faces is None:
				return None
			bx = by = bw = bh = 0
				
			for (x, y, w, h) in faces:
				if w > bw: 
				# is current face bigger than biggest face so far
					bx, by, bw, bh = x, y, w, h

			cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)
			return ((bx + bw // 2), by + (bh // 2)) 



##main code
print("starting 0OOO0 virtual 3D")
ff = FaceFinder()
#create cam
cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
	print("Could not open cam")
	exit()

while True:
  retval, frame = cap.read()
  if retval == False:
    print("camera error")

  ff.find_face(frame)
  cv2.imshow('q ', frame)

  if cv2.waitKey(30) == ord('q'):
    break


#pause = input('prees enter to end')
# destroy cam
cap.release()

cv2.destroyAllWindows()
print("Finished virtual 3D")

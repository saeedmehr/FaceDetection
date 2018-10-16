import cv2,os,argparse
#argparse is a function which is the recommended command-line parsing module in the Python standard library.
DEFAULT_OUTPUT_PATH = 'FaceCaputreImages/'
DEFAULT_CASCADE_INPUT_PATH= 'Haarcascade_frontalface_alt.xml'

 
class VideoCapture:

	def __init__(self): 
		self.count = 0
		self.argsObj = Parse()
		self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
		self.videoSource = cv2.VideoCapture(0)

	def CaptureFrames(self):
		while True:
			# Then we will give a unique number to each fram 
			frameNumber ='%08d' % (self.count)
			#Now we will get the pictures frame by frame
			ret, frame=self.videoSource.read()
			#Now we are going to set the colors to gray so that Haarcascade_frontalface can easily find the edges of the face
			screenColor= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#Customize finding the face with cascade
			faces= self.faceCascade.detectMultiScale(
			   screenColor,
			   scaleFactor = 1.1,
			   minNeighbors=5,
			   minSize=(30,30),
			  flags= cv2.CASCADE_SCALE_IMAGE)

			#Showing the final frame
			cv2.imshow('spying on you' , screenColor)

			# if the length of face is 0,then no face has been found
			if len(faces)==0:
				pass
			if len(faces) > 0:
				print('Face Detected')

			#Finding the face and then drawing a rectangle around it
				for(x,y,w,h) in faces:
					cv2.rectangle(frame,(x,y), (x+w , y+h), (0,255,0),2)

			cv2.imwrite(DEFAULT_OUTPUT_PATH + frameNumber + '.png',frame)
			self.count +=1
			if cv2.waitKey(1)==27:
				break
		
		#In the end , release the captured frames and close window
		self.videoSource.release()
		cv2.waitKey(500)
		cv2.destroyAllWindows()
		cv2.waitKey(500)




def Parse():
	parser = argparse.ArgumentParser( description='Cascade path for face detection')
	parser.add_argument('-i','--input_path' ,type = str, default = DEFAULT_CASCADE_INPUT_PATH , help='Casecade input path')
	parser.add_argument('-o','--output_path',type=str,default=DEFAULT_OUTPUT_PATH,help='Outpath path for picture taken')
	args= parser.parse_args()
	return args




def ClearImageFolder():
	if not (os.path.exists(DEFAULT_OUTPUT_PATH)):
		os.makedirs(DEFAULT_OUTPUT_PATH)

	else:
		for files in os.listdir(DEFAULT_OUTPUT_PATH):
			filePath = os.path.join(DEFAULT_OUTPUT_PATH,files)
			if os.path.isfile(filePath):
				os.unlink(filePath)

			else:
				continue
		#print "saeeeeed"
		

def main():
	ClearImageFolder()
	#Calling the class object
	faceDetectImplmentation = VideoCapture()
	 
	#Calling the captureFrame class
	faceDetectImplmentation.CaptureFrames()


if __name__=='__main__':
	main()


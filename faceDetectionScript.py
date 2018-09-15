import cv2,os,argparse

DEFAULT_OUTPUT_PATH = 'FaceCaputreImages/'
DEFAULT_CASCADE_INPUT_PATH= 'Haarcascade_frontalface_alt.xml'


	#argparse ye functionalitie ke vaghti run mikonim scripto ,parametraro betonim to comand line bezarim

class VideoCapture:

	def __init__(self): 
		self.count = 0
		self.argsObj = Parse()
		self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
		self.videoSource = cv2.VideoCapture(0)

	def CaptureFrames(self):
		while True:

			#ye adade uniq vase har frame misazim....yani ma alan 08d ke ye int hast o be self.count dadim
			frameNumber ='%08d' % (self.count)

			#hala frame by frame akso migirim

			ret, frame=self.videoSource.read()

			#hala screen color o set mikonim be gray,baraye tabe haar cascade ,ta betone edge o face o tashkhis bede

			screenColor= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#CUSTOMISE chegonegie tashkhise face tavasote cascade
			faces= self.faceCascade.detectMultiScale(
			   screenColor,
			   scaleFactor = 1.1,
			   minNeighbors=5,
			   minSize=(30,30),
			  flags= cv2.CASCADE_SCALE_IMAGE)


			#namayeshe frame nahaei

			cv2.imshow('spying on you' , screenColor)

			#agar ke length faces=0 bud ,yani hich faci peyda nashode
			if len(faces)==0:
				pass

			#agar face tashkhis dad,adadi gheir az 0 o barmigardone
			if len(faces) > 0:
				print('Face Detected')

			#face o tashkhis bede va moraba ro dore face bekesh
				for(x,y,w,h) in faces:
					cv2.rectangle(frame,(x,y), (x+w , y+h), (0,255,0),2)

			cv2.imwrite(DEFAULT_OUTPUT_PATH + frameNumber + '.png',frame)


			#AFZAYESHE COUNT vase inke har frame ye uniq adad dashte bashe

			self.count +=1
			#if esc zadim ,close mishe video
			if cv2.waitKey(1)==27:
				break
		#vaghti hamechi tamom shod, release the capture and close window
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

	#seda zadane class object
	faceDetectImplmentation = VideoCapture()
	 
	#seda zadane captureframe az class
	faceDetectImplmentation.CaptureFrames()


if __name__=='__main__':
	main()


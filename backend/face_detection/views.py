import cv2
import numpy as np
import face_recognition
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import base64
from datetime import datetime

@method_decorator(csrf_exempt, name='dispatch')
class FaceDetectionView(APIView):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.reference_image = None
        self.reference_encoding = None
        self.looking_away_start = None
        self.looking_away_threshold = 5  # seconds

    def process_frame(self, frame_data):
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Detect objects (potential electronic devices)
        objects = self.object_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Process results
        result = {
            'person_count': len(faces),
            'looking_away': False,
            'electronic_devices': len(objects) > 0,
            'same_person': True
        }
        
        # Check if person is looking away
        if len(faces) == 0:
            if self.looking_away_start is None:
                self.looking_away_start = datetime.now()
            elif (datetime.now() - self.looking_away_start).seconds >= self.looking_away_threshold:
                result['looking_away'] = True
        else:
            self.looking_away_start = None
            
            # Get face encoding for identity verification
            if self.reference_encoding is not None:
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
                    # Compare with reference encoding
                    matches = face_recognition.compare_faces([self.reference_encoding], face_encoding)
                    result['same_person'] = matches[0]
        
        return result

    def post(self, request):
        try:
            frame_data = request.data.get('frame')
            if not frame_data:
                return Response({'error': 'No frame data provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            result = self.process_frame(frame_data)
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class ReferenceImageView(APIView):
    def post(self, request):
        try:
            image_data = request.data.get('image')
            if not image_data:
                return Response({'error': 'No image data provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Decode and process reference image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Get face encoding
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return Response({'error': 'No face detected in reference image'}, 
                              status=status.HTTP_400_BAD_REQUEST)
            
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            FaceDetectionView.reference_encoding = face_encoding
            
            return Response({'message': 'Reference image set successfully'}, 
                          status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

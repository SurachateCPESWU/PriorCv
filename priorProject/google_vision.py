import io
import os
import cv2
import sys

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/TM/git/google-vision.json"

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('200dpi/400dpi_6_edit.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.text_detection(image=image)
texts = response.text_annotations


img = cv2.imread(file_name)

# sys.stdout=open(file_name + '.txt' ,"w")

for text in texts:
    textToShow = text.description
    print(textToShow)
    vertices = [(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices]
    # cv2.putText(img, textToShow, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.rectangle(img, (vertices[0][0]-10, vertices[0][1]-10), (vertices[2][0]+10, vertices[2][1]+10), (0, 255, 0), 3)

# sys.stdout.close()

# cv2.imwrite(file_name+'_output.jpg', img)
cv2.imshow('Recognize & Draw', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

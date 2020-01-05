from src import detect_faces, sliceImage_save
from PIL import Image
from io import BytesIO
import base64

def slice_Image (input_img) :
    img = Image.open(BytesIO(base64.b64decode(input_img))).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    sliceImage_save(img, bounding_boxes)

    return bounding_boxes
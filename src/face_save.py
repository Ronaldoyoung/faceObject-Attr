from PIL import ImageDraw

def sliceImage_save(img, bounding_boxes):
    img_copy = img.copy()
    # draw = ImageDraw.Draw(img_copy)
    i = ord('A')
    exp = 10
    for b in bounding_boxes:
        area = (b[0]-exp,b[1]-exp,b[2]+exp,b[3]+exp)
        img_slice = img_copy.crop(area)
        img_slice = img_slice.resize((178,218))
        img_slice.save('src/imageSave/' + str(chr(i)) + '.jpg')
        i = i + 1


    return img_copy

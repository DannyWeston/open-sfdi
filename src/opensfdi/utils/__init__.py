def centre_crop_img(img, x1, y1, x2:int = 0, y2:int = 0):
    if x2 == 0:
        x2 = img.shape[1] if x1 == 0 else -x1
        
    if y2 == 0:
        y2 = img.shape[0] if y1 == 0 else -y1
    
    return img[y1 : y2, x1 : x2]

def normalise_img(img):
    return ((img - img.min()) / (img.max() - img.min()))
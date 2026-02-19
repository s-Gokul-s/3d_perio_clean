import cv2

def enhance_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    standardized = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(standardized, None, 10, 7, 21)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

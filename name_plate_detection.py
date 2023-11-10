import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


Tk().withdraw()
file = askopenfilename()
image = cv2.imread(file)
image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_scaled = cv2.bilateralFilter(gray_scaled, 11, 17, 17)
edged = cv2.Canny(gray, 170, 200) 
cv2.imshow("Edged", edged)
cv2.waitKey(0)


contours, heirarchy  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image_1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3) 


contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
Number_Plate_Contour = 0 


for current_contour in contours:
        perimeter = cv2.arcLength(current_contour, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True) 
        if len(approx) == 4:  
            Number_Plate_Contour = approx 
            break

mask = np.zeros(gray_scaled.shape,np.uint8)
new_image1 = cv2.drawContours(mask,[Number_Plate_Contour ],0,255,-1,)
new_image1 = cv2.bitwise_and(image,image,mask=mask)
cv2.imshow("Number Plate",new_image1)
cv2.waitKey(0)
gray_scaled1 = cv2.cvtColor(new_image1, cv2.COLOR_BGR2GRAY)
ret,processed_img = cv2.threshold(np.array(gray_scaled1), 125, 255, cv2.THRESH_BINARY)
cv2.imshow("Number Plate",processed_img)
cv2.waitKey(0)
text = pytesseract.image_to_string(processed_img)
print("Number is :", text)

cv2.waitKey(0) 

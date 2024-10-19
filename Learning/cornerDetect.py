import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
img=cv2.imread('./image.png')
img = cv2.resize(img,(0,0),fx=0.75,fy=0.75)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,100,0.5,10)
corners = np.int32(corners)

for corner in corners:
    x,y=corner.ravel()
    cv2.circle(img,(x,y),5,(255,0,0),-1)
    
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])  # convert to tuple
        corner2 = tuple(corners[j][0])  # convert to tuple
        color=tuple(map(lambda x:int(x),np.random.randint(0,255,size=3)))
        cv2.line(img,corner1,corner2,color,1)
# hImg,wImg = img.size
# boxes=pytesseract.image_to_boxes(img)
# img_array = np.array(img)
# for b in boxes.splitlines():
#     x,y = corners.ravel()
#     b=b.split(' ')
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img_array,(x,hImg-y),(w,hImg-h),(0,0,255),3)


cv2.imshow('Frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

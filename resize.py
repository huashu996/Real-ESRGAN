import cv2
import os

path = os.listdir("results")

for img in path:
    img_read = cv2.imread(os.path.join("results",img))
    img_out = cv2.resize(img_read,(0,0), fx=0.25,fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite("./results/"+img,img_out)


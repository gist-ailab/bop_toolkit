import cv2
import cv2.aruco as aruco       
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)   # dictionary id
board = aruco.GridBoard_create(5, 7, 0.033, 0.004, dictionary)
img = board.draw([1000, 1500])
cv2.imwrite('board.png', img)
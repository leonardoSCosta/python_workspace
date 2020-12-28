import cv2

image = cv2.imread(
    '../../Documentos/PAIN/Images/064-ak064/ak064t1aaaff/ak064t1aaaff001.png')

print(image.shape)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

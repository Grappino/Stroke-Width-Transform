import cv2
import sys

"""
    swt Preforms stroke width transform on input image
    A novel image operator that seeks to find the value of stroke width
    for each image pixel.  It's use is meant for the task of text
    detection in natural images.
    im = RGB input image of size m x n x 3
    searchDirection = gradient direction is either 1 to detect dark text on light
    background or -1 to detect light text on dark background.
    swtMap = resulting mapping of stroke withs for image pixels
"""


def swt():
    argc = len(sys.argv)
    if argc > 1:
        image = cv2.imread(sys.argv[1], 0)
    else:
        print "Errore! Nessuna immagine inserita!"
    #We use the Canny Edge Detection to find the edges of the image
    edgeMap = cv2.Canny(image, 100, 300)
    cv2.imshow('edgeMap ', edgeMap)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    swt()


if __name__ == "__main__":
    main()


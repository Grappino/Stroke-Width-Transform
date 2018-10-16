import cv2
import sys
import numpy as np

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


# 3.1 The Stroke Width Transform
def swt():
    argc = len(sys.argv)
    if argc > 1:
        image = cv2.imread(sys.argv[1], 0)
    else:
        print "Errore! Nessuna immagine inserita!"
    # We use the Canny Edge Detection to find the edges of the image
    edge_map = cv2.Canny(image, 100, 300)
    """
    # testing the Canny Edge
    cv2.imshow('edge_map', edge_map)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """
    # SWT Map with all pixel initialized with infinite as swt value
    swt_map = 255*np.ones(image.shape, image.dtype)
    # row, column
    height, width = image.shape
    # x gradient, y-gradient are computed using Sobel operator
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=-1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=-1)
    # theta
    theta = np.arctan2(gy, gx)
    # list of the rays found
    rays = []
    # the cosine and the sine of the gradient angle represent the basic increment for passing
    # from one point of the radius to the next
    cos_g, sin_g = np.cos(theta), np.sin(theta)
    for i in range(height):
        for j in range(width):
            # if the current point is and edge there is a candidate ray
            if edge_map[i, j]:
                # candidate ray for current stroke width
                ray = [(i, j)]
                # cur_i, cur_j : coordinates of the previous pixel
                # cnt : iteration counter
                cur_i, cur_j, cnt = i, j, 0
                while True:
                    cnt += 1
                    # coordinates of the new current pixel
                    next_i = int(np.floor(i + sin_g[i, j] * cnt))
                    next_j = int(np.floor(j + cos_g[i, j] * cnt))
                    # if the new coordinates are within the limits of the image
                    if next_i < 0 or next_i >= height or next_j < 0 or next_j >= width:
                        break
                    #  if the new point is different then the older point
                    if next_i != cur_i or next_j != cur_j:
                        # add the point to the ray
                        ray.append((next_i, next_j))
                        # if the new point is an edge then the candidate ray ends
                        if edge_map[next_i, next_j]:
                            # a radius is valid if the angle of the gradient at the starting point is approximately
                            # opposite the angle of the gradient at the end point
                            sum_theta = np.abs(theta[cur_i, cur_j] + theta[next_i, next_j])
                            # we need to test more values like pi/2, pi/3, ...
                            if sum_theta <= np.pi/6:
                                # the width of the current stoke is the distance between the start and end points
                                stroke_width = np.sqrt(np.power((next_i - i), 2) + np.power((next_j - j), 2))
                                for (_i, _j) in ray:
                                    # assign the value to each pixel of the ray if it did not have a smaller value
                                    swt_map[_i, _j] = min(swt_map[_i, _j], stroke_width)
                                # add the rays to the list
                                rays.append(ray)
                                break
                    # update previous coordinates
                    cur_i, cur_j = next_i, next_j
    # np.median():
    # Given a vector V of length N,
    # the median of V is the middle value of a sorted copy of V, V_sorted
    # V_sorted[(N-1)/2], when N is odd, and the average of the two middle values of V_sorted when N is even

    for ray in rays:
        # assign to each pixel in a ray the median swt value of pixels in that ray
        # if less than the previous value
        median = np.median([swt_map[i, j] for (i, j) in ray])
        for (i, j) in ray:
            swt_map[i, j] = min(median, swt_map[i, j])

    """
    # just a test part to see if the swt works
    cv2.imshow('swt_map', swt_map)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    return swt_map


# 3.2 Finding letters candidates
def letters_candidates(swt_map):
    # labels_map initialized to 0
    labels_map = np.zeros(swt_map.shape)
    # strokes_candidate
    strokes_candidate = []
    # number of rows and columns of swt map
    nr, nc = swt_map.shape
    # first valid label and region
    label = 1
    for i in range(nr):
        for j in range(nc):
            # if the current pixel is in a stroke
            # assign it to a region with the current label
            # search ... for similar swt value
            if 255 > swt_map[i, j] > 0 and labels_map[i, j] == 0:
                stroke_candidate = [(i, j)]
                point_list = [(i, j)]
                labels_map[i, j] = label
                while len(point_list) > 0:
                    pi, pj = point_list.pop(0)
                    for ni in range(max(pi - 1, 0), min(pi + 2, nr - 1)):
                        for nj in range(max(pj - 1, 0), min(pj + 2, nc - 1)):
                            if 255 > swt_map[ni, nj] > 0 and labels_map[ni, nj] == 0:
                                if 0.333 < swt_map[ni, nj] / swt_map[i, j] < 3.0:
                                    labels_map[ni, nj] = label
                                    point_list.append((ni, nj))
                                    stroke_candidate.append((ni, nj))
                label += 1
                strokes_candidate.append(stroke_candidate)
    """
    for x in strokes_candidate:
        print x
    """
    letters = []
    # now we check the variance of the possible strokes and we reject the area with too high variance(half of the mean)
    for stroke in strokes_candidate:
        swt_vector = []
        for point in stroke:
            swt_vector.append(swt_map[point[0], point[1]])
        if np.var(swt_vector) <= 0.5 * np.mean(swt_vector):
            # we search now the min and max value of x and y in the stroke
            max_x, min_x, max_y, min_y = 0, nc, 0, nr
            for point in stroke:
                if point[0] > max_x:
                    max_x = point[0]
                if point[0] < min_x:
                    min_x = point[0]
                if point[1] > max_y:
                    max_y = point[1]
                if point[1] < min_y:
                    min_y = point[1]
            s_width = max_x - min_x
            s_height = max_y - min_y
            if min_x == max_x:
                hw_ratio = 11
            else:
                hw_ratio = s_height / s_width
            if min_y == max_y:
                wh_ratio = 11
            else:
                wh_ratio = s_width / s_height
            # we check that the aspect_ratio is a value between 0.1 and 10
            if hw_ratio <= 10 and wh_ratio <= 10:
                # the ratio between the diameter of connected components and its median stroke
                # must be a value less then 10
                diam = np.sqrt(np.power(s_width, 2) + np.power(s_height, 2))
                med = np.median(swt_vector)
                dm_ratio = diam / med
                if dm_ratio <= 10:
                    # we check that the height is a value between 10px and 300px
                    if 10 <= s_height <= 300:
                        letters.append(stroke)

    result = np.zeros(swt_map.shape)
    for l in letters:
        for p in l:
            result[p[0]][p[1]] = 255
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    swt_map = swt()
    letters_candidates(swt_map)


if __name__ == "__main__":
    main()


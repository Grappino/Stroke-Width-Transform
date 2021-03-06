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
def swt_transform(img, edges, clear_text_on_dark_background=True):
    nr, nc = img.shape
    # Canny Edge detections
    edges = edges.astype(np.bool)
    # x gradient, y-gradient are computed using Sobel operator
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    # the gradient orientation (theta) can be estimated as atan(gy/gx)
    theta = np.arctan2(gy, gx)
    # swt map has the same dimension of the image
    swt = np.zeros(img.shape, img.dtype)
    # the initial stroke width of each pixel is infinity(255)
    swt[:] = 255
    # list of the rays found
    rays = []
    # the cosine and the sine of the gradient angle represent the basic increment for passing
    # from one point of the radius to the next
    cos_g, sin_g = np.cos(theta), np.sin(theta)
    for i in range(nr):
        for j in range(nc):
            # if the current point is and edge there is a candidate ray
            if edges[i, j]:
                # candidate ray for current stroke width
                ray = [(i, j)]
                # cur_i, cur_j : coordinates of the previous pixel
                # cnt : iteration counter
                cur_i, cur_j, cnt = i, j, 0
                while True:
                    cnt += 1
                    # coordinates of the new current pixel
                    # the next_i and next_j need to -cnt if we are in ahe case of dark text on light background
                    if clear_text_on_dark_background:
                        next_i = int(np.floor(i + sin_g[i, j] * cnt))
                        next_j = int(np.floor(j + cos_g[i, j] * cnt))
                    else:
                        next_i = int(np.floor(i + sin_g[i, j] * -cnt))
                        next_j = int(np.floor(j + cos_g[i, j] * -cnt))
                    # if the new coordinates are within the limits of the image
                    if next_i < 0 or next_i >= nr or next_j < 0 or next_j >= nc:
                        break
                    #  if the new point is inside the image
                    if next_i != cur_i or next_j != cur_j:
                        # if the new point is an edge then the candidate ray ends
                        if edges[next_i, next_j]:
                            ray.append((next_i, next_j))
                            # a radius is valid if the angle of the gradient at the starting point is approximately
                            # opposite the angle of the gradient at the end point
                            v = np.abs(np.abs(theta[i, j] - theta[next_i, next_j]) - np.pi)
                            if v < np.pi/2:
                                # the width of the current stoke is the distance between the start and end points
                                width = np.sqrt(np.power((next_i - i), 2) + np.power((next_j - j), 2))
                                for (_i, _j) in ray:
                                    # assign the value to each pixel of the ray if it did not have a smaller value
                                    swt[_i, _j] = min(swt[_i, _j], width)
                                # add the rays to the list
                                rays.append(ray)
                            break
                        # if the new point is not an edge then add the point to the ray
                        ray.append((next_i, next_j))
                    # update previous coordinates
                    cur_i, cur_j = next_i, next_j

    # np.median():
    # Given a vector V of length N,
    # the median of V is the middle value of a sorted copy of V, V_sorted
    # V_sorted[(N-1)/2], when N is odd, and the average of the two middle values of V_sorted when N is even
    for ray in rays:
        # assign to each pixel in a ray the median swt value of pixels in that ray
        # if less than the previous value
        median = np.median([swt[i, j] for (i, j) in ray])
        for (i, j) in ray:
            swt[i, j] = min(median, swt[i, j])

    # we filter the swt to eliminate some errors: lines that connect wrong borders
    for i in range(nr):
        for j in range(nc):
            if swt[i][j] != 255:
                count, neighborhood = 0, []
                neighborhood.append((i - 1, j - 1))
                neighborhood.append((i - 1, j + 1))
                neighborhood.append((i + 1, j - 1))
                neighborhood.append((i + 1, j + 1))
                neighborhood.append((i, j - 1))
                neighborhood.append((i - 1, j))
                neighborhood.append((i + 1, j))
                neighborhood.append((i, j + 1))
                for n in neighborhood:
                    if 0 <= n[0] < nr and 0 <= n[1] < nc:
                        if swt[n[0]][n[1]] != 255:
                            count += 1
                if count <= 2:
                    swt[i][j] = 255
    """
    # test to see if the swt works
    cv2.imshow('swt', swt)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    return swt


# 3.2 Finding letters candidates
def letters_finder(swt, edge_map):
    # We start using a modified version of Connected Component algorithm
    # labels map initialized to 0
    labels = np.zeros(swt.shape, swt.dtype)
    # layers list
    strokes = []
    # number of rows and columns of swt map
    nr, nc = swt.shape
    # first valid label and region
    label = 1
    for i in range(nr):
        for j in range(nc):
            # if the current pixel is in a stroke
            # assign it to a region with the current label
            # search ... for similar swt value
            if edge_map[i, j] and 255 > swt[i, j] > 0 and labels[i, j] == 0:
                # list of the point in the current region
                point_list = [(i, j)]
                labels[i, j] = label
                # searching for similar swt value
                while len(point_list) > 0:
                    pi, pj = point_list.pop(0)
                    neighborhood = []
                    neighborhood.append((pi - 1, pj - 1))
                    neighborhood.append((pi - 1, pj + 1))
                    neighborhood.append((pi + 1, pj - 1))
                    neighborhood.append((pi + 1, pj + 1))
                    neighborhood.append((pi, pj - 1))
                    neighborhood.append((pi - 1, pj))
                    neighborhood.append((pi + 1, pj))
                    neighborhood.append((pi, pj + 1))
                    for n in neighborhood:
                        if 0 <= n[0] < nr and 0 <= n[1] < nc:
                            if labels[n[0], n[1]] == 0 and 255 > swt[n[0], n[1]] > 0:
                                if 0.333 < swt[n[0], n[1]] / swt[i, j] < 3.0:
                                    labels[n[0], n[1]] = label
                                    point_list.append((n[0], n[1]))
                # pass to the next label and region
                label += 1
    """
    # test to see the different labels
    cv2.imshow('labels', labels)
    cv2.waitKey()
    """

    # No we need to rectify the problem that we have multiple labels for connected regions.
    for i in range(nr):
        for j in range(nc):
            if labels[i, j] != 0:
                same_label, neighborhood = [], []
                neighborhood.append((i - 1, j - 1))
                neighborhood.append((i - 1, j + 1))
                neighborhood.append((i + 1, j - 1))
                neighborhood.append((i + 1, j + 1))
                neighborhood.append((i, j - 1))
                neighborhood.append((i - 1, j))
                neighborhood.append((i + 1, j))
                neighborhood.append((i, j + 1))
                # at the beginning we search for the labels that are in the same letter
                for n in neighborhood:
                    if nr > n[0] >= 0 and nc > n[1] >= 0 and labels[n[0], n[1]] != 0:
                        if labels[i, j] != labels[n[0], n[1]]:
                            same_label.append(labels[n[0], n[1]])
                # then we assign the same labels to all the pixels that have the same labels found in the neighborhood
                for to_fix in same_label:
                    for x in range(nr):
                        for y in range(nc):
                            if labels[x, y] == to_fix:
                                labels[x, y] = labels[i, j]
    """
    # second test to see the new labels
    cv2.imshow('labels_fixed', labels)
    cv2.waitKey()
    """

    # now we assign the points with the same labels to the same group( a letter candidate)
    for label in np.unique(labels):
        if label != 0.0:
            stroke = []
            for i in range(nr):
                for j in range(nc):
                    if labels[i, j] == label:
                        stroke.append((i, j))
            strokes.append(stroke)
    # end of the Connected Component algorithm

    """ 
    # now we print the single strokes, letters candidates
    for s in strokes:
        temp = np.zeros(swt.shape)
        for p in s:
            temp[p[0], p[1]] = 255
        cv2.imshow('temp', temp)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """

    letters = []
    # now we check the variance of the possible strokes and we reject the area with too high variance
    for stroke in strokes:
        var = np.var([swt[i, j] for (i, j) in stroke])
        mean = np.mean([swt[i, j] for (i, j) in stroke])
        if var <= 2 * mean:
            # we search now the min and max value of x and y in the stroke
            s_width = get_letter_width(stroke)
            s_height = get_letter_height(stroke)
            if s_height == 0 or s_width == 0:
                hw_ratio = 11
                wh_ratio = 11
            else:
                hw_ratio = s_height / s_width
                wh_ratio = s_width / s_height
            # we check that the aspect_ratio is a value between 0.1 and 10
            if hw_ratio <= 10 and wh_ratio <= 10:
                # the ratio between the diameter of connected components and its median stroke
                # must be a value less then 10
                diam = np.sqrt(np.power(s_width, 2) + np.power(s_height, 2))
                med = get_letter_swt(stroke, swt)
                dm_ratio = diam / med
                if dm_ratio <= 10:
                    # we check that the height is a value between 10px and 300px
                    if 10 <= s_height <= 300:
                        letters.append(stroke)

    # now we print the different letters
    """
    for label in letters:
        print "Height: " + str(get_letter_height(label)) + "px"
        print "Width: " + str(get_letter_width(label)) + "px"
        print "SWT: " + str(get_letter_swt(label, swt))
        print get_letter_extreme_sx_dx(label)
        print get_letter_extreme_top_down(label)
        temp = np.zeros(swt.shape)

        for p in label:
            temp[p[0], p[1]] = 255
        cv2.imshow('temp', temp)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    return letters


def get_letter_height(letter):
    # difference between the highest and the lowest value of y
    y, x = letter[0]
    y_min, y_max = y, y
    for point in letter:
        if point[0] > y_max:
            y_max = point[0]
        elif point[0] < y_min:
            y_min = point[0]
    return abs(y_min - y_max)


def get_letter_width(letter):
    # difference between the highest and the lowest value of x
    y, x = letter[0]
    x_min, x_max = x, x
    for point in letter:
        if point[1] > x_max:
            x_max = point[1]
        elif point[1] < x_min:
            x_min = point[1]
    return abs(x_min - x_max)


def get_letter_swt(letter, swt):
    # medium value of swt of all internal points
    return np.median([swt[i, j] for (i, j) in letter])


def get_letter_extreme_sx_dx(letter):
    # return the most extern px value on right and left side of a letter
    # we use these two values to calculate the distance between two letters
    y, x = letter[0]
    x_min, x_max = x, x
    for point in letter:
        if point[1] > x_max:
            x_max = point[1]
        elif point[1] < x_min:
            x_min = point[1]
    return x_min, x_max


def get_letter_extreme_top_down(letter):
    # return the most extern px value on top and down side of a letter
    y, x = letter[0]
    y_min, y_max = y, y
    for point in letter:
        if point[0] > y_max:
            y_max = point[0]
        elif point[0] < y_min:
            y_min = point[0]
    return y_min, y_max


def highlight_letters(letters, original, clear_text_on_dark_background=True):
    image = original.copy()
    for letter in letters:
        x_min, x_max = get_letter_extreme_sx_dx(letter)
        y_min, y_max = get_letter_extreme_top_down(letter)
        for i in range(y_min, y_max + 1):
            if clear_text_on_dark_background:
                image[i][x_min] = 255
                image[i][x_max] = 255
            else:
                image[i][x_min] = 0
                image[i][x_max] = 0
        for i in range(x_min, x_max + 1):
            if clear_text_on_dark_background:
                image[y_min][i] = 255
                image[y_max][i] = 255
            else:
                image[y_min][i] = 0
                image[y_max][i] = 0
    cv2.imshow("letters_image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def highlight_words(words, original, clear_text_on_dark_background=True):
    image = original.copy()
    for word in words:
        y_min_group, y_max_group = np.Infinity, 0
        x_max_group, x_min_group = 0, np.Infinity
        for letter in word:
            x_min, x_max = get_letter_extreme_sx_dx(letter)
            y_min, y_max = get_letter_extreme_top_down(letter)
            if x_min < x_min_group:
                x_min_group = x_min
            if x_max > x_max_group:
                x_max_group = x_max
            if y_min < y_min_group:
                y_min_group = y_min
            if y_max > y_max_group:
                y_max_group = y_max
        for i in range(y_min_group, y_max_group + 1):
            if clear_text_on_dark_background:
                image[i][x_min_group] = 255
                image[i][x_max_group] = 255
            else:
                image[i][x_min_group] = 0
                image[i][x_max_group] = 0
        for i in range(x_min_group, x_max_group + 1):
            if clear_text_on_dark_background:
                image[y_min_group][i] = 255
                image[y_max_group][i] = 255
            else:
                image[y_min_group][i] = 0
                image[y_max_group][i] = 0
    cv2.imshow("final_image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 3.3 Grouping letters into text line
def words_finder(letters, swt):
    # now we need to find the letters that form a single word
    # the letters can be disordered in the letters vector (e.g "puma" -> p,m,u,a with puma_logo)
    # as first thing i need to reunion the words that have similar dimension
    # then we need to analyze the groups and split every groups in new sub-groups tha have similar swt and are nearby
    labels = np.zeros(len(letters))
    label = 0
    # the counter is used to moving through the labels vector
    cnt = 0
    for first_letter in letters:
        if labels[cnt] == 0:
            label += 1
            labels[cnt] = label
        # we grab the first point of a letter to check if the next letter is not the same letter
        fp_fl = first_letter[0]
        count = 0
        for second_letter in letters:
            fp_sl = second_letter[0]
            if fp_fl != fp_sl:
                # letters on the same word have the ratio between the median stroke widths has to be less than 2.0
                if 0.5 < get_letter_swt(first_letter, swt)/get_letter_swt(second_letter, swt) < 2:
                    # we control that we are not considering a letter that is up or down another one
                    yf_min, yf_max = get_letter_extreme_top_down(first_letter)
                    ys_min, ys_max = get_letter_extreme_top_down(second_letter)
                    height_fl = get_letter_height(first_letter)
                    height_sl = get_letter_height(second_letter)
                    height_to_use = height_fl
                    if height_fl < height_sl:
                        height_to_use = height_sl
                    if abs(yf_max - ys_max) < height_to_use/2:
                        # the height ratio of the letters must not exceed 2.0 (due to the difference between capital and
                        # lower case letters)
                        if 0.5 < get_letter_height(first_letter)/get_letter_height(second_letter) < 2:
                            # now we control the distance between the letters that must be at most 1/3
                            # of the width of the larger letter
                            xf_min, xf_max = get_letter_extreme_sx_dx(first_letter)
                            xs_min, xs_max = get_letter_extreme_sx_dx(second_letter)
                            width_fl = get_letter_width(first_letter)
                            width_sl = get_letter_width(second_letter)
                            width_to_use = width_fl
                            if width_fl < width_sl:
                                width_to_use = width_sl
                            if abs(xf_max - xs_min) < width_to_use/3 or abs(xs_max - xf_min) < width_to_use/3:
                                if labels[count] == 0:
                                    labels[count] = labels[cnt]
                                elif labels[count] < labels[cnt]:
                                    temp = labels[cnt]
                                    for i in range(len(labels)):
                                        if labels[i] == temp:
                                            labels[i] = labels[count]
                                else:
                                    temp2 = labels[count]
                                    for i in range(len(labels)):
                                        if labels[i] == temp2:
                                            labels[i] = labels[cnt]
            count += 1
        cnt += 1
        # control on the assignment of the labels
        # print labels

    # we need to aggregate the letters with the same labels in a single word
    words = []
    for lab in np.unique(labels):
        cnt = 0
        word = []
        for letter in letters:
            if labels[cnt] == lab:
                word.append(letter)
            cnt += 1
        # this condition is suggested in the official documentation but we are not using that
        # if len(words) >= 3:
        words.append(word)
    # we check number of words in the image
    # print len(words)
    return words


def main():
    argc = len(sys.argv)
    if argc > 1:
        image = cv2.imread(sys.argv[1], 0)
    else:
        print "Errore! Nessuna immagine inserita!"
    # We use the Canny Edge Detection to find the edges of the image
    edge_map = cv2.Canny(image, 100, 300)
    # testing the Canny Edge
    """
    cv2.imshow('edge_map', edge_map)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """
    # setting the third parameter of swt as False := we consider a dark text on light background
    if argc > 2:
        swt_map = swt_transform(image, edge_map, False)
    else:
        swt_map = swt_transform(image, edge_map)
    letters = letters_finder(swt_map, edge_map)
    if argc > 2:
        highlight_letters(letters, image, False)
    else:
        highlight_letters(letters, image)
    words = words_finder(letters, swt_map)
    if argc > 2:
        highlight_words(words, image, False)
    else:
        highlight_words(words, image)


if __name__ == "__main__":
    main()

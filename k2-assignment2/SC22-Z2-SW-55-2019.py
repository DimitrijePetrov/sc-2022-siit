import sys
import cv2
import pandas
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    img_bin = np.ndarray((height, width), dtype=np.uint8)
    img_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 187, 3)
    return img_bin


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train, epochs):
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)

    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def merge_regions_into_letters(regions_array, img_bin):
    letters = [regions_array[0]]

    for r_i in range(1, len(regions_array)):
        x, y, w, h = regions_array[r_i][1]
        x_, y_, w_, h_ = regions_array[r_i - 1][1]

        if y_ > y + h and x > x_ and (x + w) < (x_ + w_):  # merges dots with i-s
            letters.pop(-1)
            i_region = img_bin[y:y_ + h_ + 1, x_:x_ + w_ + 1]
            letters.append([resize_region(i_region), (x_, y, w_, y_ + h_ - y)])

        elif not (x_ <= x and (x + w) <= (x_ + w_) and y_ <= y and (y + h) <= (y_ + h_)):  # removes holes in letters
            letters.append(regions_array[r_i])

    return letters


def draw_regions(image_orig, regions_array):  # draws all regions on image
    for region in regions_array:
        x, y, w, h = region[1]
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_orig


def display_result(outputs, alphabet):  # retturns letters as a string
    result = ""
    for output in outputs:
        result += alphabet[winner(output)]
    return result


def select_roi_with_distances(image_orig, img_bin):
    contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (1600 > area > 200 and h > 30 and w > 15) or (15 > h > 9 and 15 > w > 9):
            region = img_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda r: r[1][0])
    regions_array = merge_regions_into_letters(regions_array, img_bin)
    image_orig = draw_regions(image_orig, regions_array)

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]

    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # x_next - (x_current + w_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def display_result_with_spaces(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result


def hamming_distance(string1, string2):  # calculates Hamming distance of two strings
    distance = 0
    shorter = string1 if len(string1) < len(string2) else string2
    longer = string2 if len(string1) < len(string2) else string1
    for i in range(len(shorter)):
        if shorter[i] != longer[i]:
            distance += 1
    distance += len(longer) - len(shorter)
    return distance


folder_path = sys.argv[1]
alphabet = [
    'A', 'r', 'b', 'e', 'i', 't', 's', 'p', 'l', 'a', 'z', 'E', 'P', 'f', 'o', 'g', 'ss', 'L', 'n', 'w', 'M', 'k', 'S',
    'U', 'm', 'c', 'h', 'V', 'd', 'u', 'W',
]
all_letters = []

image_color = load_image(folder_path + '/a1.jpg')
img = image_bin(image_gray(load_image(folder_path + '/a1.jpg')))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
letters.pop(10)
all_letters.extend(letters)

image_color = load_image(folder_path + '/e1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0], letters[6]])

image_color = load_image(folder_path + '/e2.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[2], letters[3], letters[5], letters[11]])

image_color = load_image(folder_path + '/l1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0], letters[4], letters[6]])

image_color = load_image(folder_path + '/m1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0], letters[3]])

image_color = load_image(folder_path + '/s1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0]])

image_color = load_image(folder_path + '/u1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0], letters[1], letters[3], letters[4]])

image_color = load_image(folder_path + '/v1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0], letters[6], letters[7]])

image_color = load_image(folder_path + '/w1.jpg')
img = image_bin(image_gray(image_color))
_, letters, _ = select_roi_with_distances(image_color.copy(), img)
all_letters.extend([letters[0]])


inputs = prepare_for_ann(all_letters)
outputs = convert_output(alphabet)
ann = create_ann(output_size=len(all_letters))
ann = train_ann(ann, inputs, outputs, epochs=2500)


df = pandas.read_csv(folder_path + '/res.txt', sep=", ", engine="python")
file_names = df['file'].to_list()
correct_solutions = df['text'].to_list()

mae = 0
for i in range(len(file_names)):
    image_color = load_image(folder_path + '/' + file_names[i])
    img = image_bin(image_gray(image_color))
    selected_regions, letters, distances = select_roi_with_distances(image_color.copy(), img)
    # display_image(selected_regions, True)

    if file_names[i] == 'e1.jpg':
        distances = np.array(distances).reshape(len(distances), 1)
        k_means = KMeans(n_clusters=2)
        k_means.fit(distances)
        inputs = prepare_for_ann(letters)
        result = ann.predict(np.array(inputs, np.float32))
        solution = display_result_with_spaces(result, alphabet, k_means)
    else:
        inputs = prepare_for_ann(letters)
        result = ann.predict(np.array(inputs, np.float32))
        solution = display_result(result, alphabet)

    mae += hamming_distance(solution, correct_solutions[i])
    print("{} - {} - {}".format(file_names[i], correct_solutions[i], solution))

print(mae)


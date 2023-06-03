import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from keras_facenet import FaceNet
import tkinter as tk
import os

# Special Files
INPUT_DIRECTORY = os.path.join(os.getcwd(), "input")
OUTPUT_DIRECTORY = os.path.join(os.getcwd(), "output")
RESOURCES_DIRECTORY = os.path.join(os.getcwd(), "resources")
NO_FACE_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "-faces")

EXEMPLARS_DIRECTORY = os.path.join(os.getcwd(), "exemplars")
EXEMPLARS_FACES = os.path.join(EXEMPLARS_DIRECTORY, "faces")
EXEMPLARS_DATABASE = os.path.join(EXEMPLARS_DIRECTORY, "exemplars.json")
NAME_MAP_DATABASE = os.path.join(EXEMPLARS_DIRECTORY, "names.json")

# Load Models
# YuNet overview and 'tutorial' could be found here:
# https://medium.com/@silkworm/yunet-ultra-high-performance-face-detection-in-opencv-a-good-solution-for-real-time-poc-b01063e251d5
# Load face detector
print("Loading YuNet...")
weights = os.path.join(RESOURCES_DIRECTORY, "face_detection_yunet_2022mar.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
print("YuNet loading complete!")
# Load the FaceNet model
# FaceNet code inspiration could be found here:
# https://www.youtube.com/watch?v=_CfhRzAlHQM
print("Loading FaceNet...")
face_net = FaceNet()
print("FaceNet loading complete!")


# Preset calculations
# Calculate preferred size to fit image to screen
root = tk.Tk()
FIT_SCREEN_SIZE = (root.winfo_screenheight() * 0.5, root.winfo_screenwidth() * 0.5)


def ratio_resize(image, new_size=None):
    if new_size is None:
        new_size = FIT_SCREEN_SIZE

    # Use the side with the smallest difference with the original
    difference = tuple(map(lambda i, j: (i - j) * -((i - j) < 0), image.shape, new_size))
    side = difference.index(min(difference))
    # Identify ratio change
    from_side, to_side = image.shape[side], new_size[side]
    percentage = to_side / from_side
    # Resize image
    new_1 = [int(i * percentage) for i in list(image.shape)]
    del new_1[2]
    a, b = tuple(new_1)
    brand_new_size = (b, a)

    return cv2.resize(image, brand_new_size, interpolation=cv2.INTER_AREA)


def get_face_area(face):
    x, y, w, h = list(map(int, face[:4]))
    return w*h


def get_face_distance(face, image_height, image_width):
    special_points = []
    # Center point
    special_points.append([image_width/2, image_height/2])
    # Rule of thirds intersections
    third_x = image_width/3
    third_y = image_height/3
    for x in range(1, 3):
        for y in range(1, 3):
            special_points.append([x*third_x, y*third_y])

    # Return smallest distance to special point
    x, y, w, h = list(map(int, face[:4]))
    face_center_x = x + (w/2)
    face_center_y = y + (h/2)
    return np.min(euclidean_distances(special_points, [[face_center_x, face_center_y]]))


def extract_faces(img, show_landmarks=False, show_crop=False):
    local_img = img.copy()

    # Set input size
    height, width, _ = local_img.shape
    face_detector.setInputSize((width, height))

    # Detect faces
    _, faces = face_detector.detect(local_img)
    faces = faces if faces is not None else []

    if show_landmarks:
        mark_image(local_img, faces)
        # Display img
        cv2.imshow("face detection", local_img)
        cv2.waitKey(0)

    if show_crop:
        for face in faces:
            box = list(map(int, face[:4]))
            x, y, w, h = box
            crop_img = local_img[y+1:y+h-1, x+1:x+w-1]
            # Display img
            cv2.imshow("face detection", crop_img)
            cv2.waitKey(0)

    # To clear previous images
    # cv2.destroyWindow("face detection")

    return faces


def mark_image(img, faces):
    # Add bounding boxes and landmarks
    for face in faces:
        # Bounding box
        box = list(map(int, face[:4]))
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(img, box, color, thickness, cv2.LINE_AA)

        # Landmarks (Right Eye, Left Eye, Nose, Right Mouth Corner, Left Mouth Corner)
        landmarks = list(map(int, face[4:len(face)-1]))
        landmarks = np.array_split(landmarks, len(landmarks) / 2)
        for landmark in landmarks:
            radius = 5
            thickness = -1
            cv2.circle(img, landmark, radius, color, thickness, cv2.LINE_AA)

        # Confidence
        confidence = face[-1]
        confidence = "{:.2f}".format(confidence)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        cv2.putText(img, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

    return img


def embed_images(imgs):
    signatures = face_net.embeddings(imgs)
    return signatures


def cluster_embeddings(embeddings_scaled, similarity_function=None, graph=False):
    if similarity_function is None:
        similarity_function = np.min

    # Compute euclidean distance for preference
    # Inspired: https://stackoverflow.com/questions/33187354/affinity-propagation-preferences-initialization
    similarity_matrix = -(euclidean_distances(embeddings_scaled, embeddings_scaled)**2)
    similarity = similarity_function(similarity_matrix)

    # Apply Affinity Propagation clustering
    ap = AffinityPropagation(preference=similarity)
    ap.fit(embeddings_scaled)

    # Retrieve cluster labels and exemplars
    cluster_labels = ap.labels_
    exemplars = ap.cluster_centers_indices_

    if graph:
        graph_embeddings(embeddings_scaled, cluster_labels, exemplars)
    return (cluster_labels, exemplars)


def graph_embeddings(embeddings_scaled, cluster_labels, exemplars):
    # Reduce dimensionality while maintaining distance using TSNE
    # To allow visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    embeddings_2d = tsne.fit_transform(embeddings_scaled)

    # Plot the data points with cluster colors
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')
    plt.scatter(embeddings_2d[exemplars, 0], embeddings_2d[exemplars, 1], marker='x', color='red', s=100, label='Exemplars')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.title('Affinity Propagation Clustering')
    plt.show()






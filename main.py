import cv2
import os
import shutil
from sklearn.preprocessing import StandardScaler
import json
from helpers import *


# Initialize Exemplars Database and Name Map
if not os.path.exists(EXEMPLARS_DATABASE):
    with open(EXEMPLARS_DATABASE, "w") as f:
        json.dump({}, f)

if not os.path.exists(NAME_MAP_DATABASE):
    with open(NAME_MAP_DATABASE, "w") as f:
        json.dump({}, f)


def main():
    # Universal index mapping (one index to many lists)
    # Things to be stores
    image_embeddings = [] # To be set later
    main_face_imgs = []
    image_names = []
    # Loop all images in input folder
    for image_name in os.listdir(INPUT_DIRECTORY):
        print(f"Processing {image_name}...")
        # Open image
        image_path = os.path.join(INPUT_DIRECTORY, image_name)
        img = cv2.imread(image_path)
        img = ratio_resize(img)

        # Identify main face
        main_face = None
        # Extract faces
        faces = extract_faces(img)
        # Set main face as the face with the highest confidence (NOTE: is it the first element?)
        # TODO: Include the face heuristic that the closer the face is to the center (or somewhere special)
        # Increase its likeliness to be main face
        for face in faces:
            confidence = face[-1]
            if main_face is None or confidence > main_face[-1]:
                main_face = face

        # No face
        if main_face is None:
            if not os.path.exists(NO_FACE_DIRECTORY):
                os.mkdir(NO_FACE_DIRECTORY)

            destination = os.path.join(NO_FACE_DIRECTORY, image_name)

            try:
                os.rename(image_path, destination)
            except WindowsError:
                os.remove(destination)
                os.rename(image_path, destination)

            continue

        # Crop main_face
        main_face_rect = list(map(int, main_face[:4]))
        x, y, w, h = main_face_rect
        main_face_img = img[y+1:y+h-1, x+1:x+w-1]

        # Preprocess the image for FaceNet
        main_face_img = cv2.cvtColor(main_face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        main_face_img = cv2.resize(main_face_img, (160, 160))  # Resize image to match FaceNet's input size

        # Store main_face_img
        main_face_imgs.append(main_face_img)
        # Store image_name
        image_names.append(image_name)        

    # Create and store main_face embeddings from main_face_imgs
    image_embeddings = np.array(embed_images(np.array(main_face_imgs)))
    image_names = np.array(image_names)


    # Cluster embeddings
    # RUN 1 (New images)
    # Preprocess signatures
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(image_embeddings)

    new_cluster_labels, new_exemplars = cluster_embeddings(embeddings_scaled)
    print(f"Number of clusters from new data: {len(new_exemplars)}")
    # End of RUN 1
    # Everything after this point is chaos TT (send help)


    # Get exemplars to be clustered (NOTE: These are already scaled embeddings)
    with open(EXEMPLARS_DATABASE, "r") as f:
        exemplars_data = json.load(f)

    # Get old exemplar data
    old_exemplars_cluster_labels = list(exemplars_data.keys())
    old_exemplars_embeddings = list(exemplars_data.values())
    
    # Get new exemplar data
    new_exemplars_embeddings = embeddings_scaled[new_exemplars].tolist()
    new_exemplars_names = image_names[new_exemplars]


    # RUN 2 (Exemplars)
    # Combine exemplars to cluster
    combined_exemplars_embeddings = np.array(old_exemplars_embeddings + new_exemplars_embeddings)
    # Tried my best to only join exemplars of same face
    combined_cluster_labels, combined_exemplars = cluster_embeddings(combined_exemplars_embeddings, similarity_function=lambda x: np.mean(x) + np.std(x))
    print(f"Number of clusters from combined data: {len(combined_exemplars)}")
    # End of RUN 2


    # New cluster name assignment of old exemplars
    old_combined_cluster = list(map(str, combined_cluster_labels[:len(old_exemplars_embeddings)]))
    # New cluster name assignment of new exemplars
    new_combined_cluster = list(map(str, combined_cluster_labels[len(old_exemplars_embeddings):]))
    # Map new cluster name assignment to correspond to the database naming
    # Heavily assumes that no old exemplars are combined to form one cluster
    combined2database_label = dict(zip(old_combined_cluster, old_exemplars_cluster_labels))

    # Update Exemplars Database
    for cluster_label, exemplar_embedding in zip(combined_cluster_labels[combined_exemplars], combined_exemplars_embeddings[combined_exemplars]):
        # If exemplar is not part of the old database, make a new entry
        if str(cluster_label) not in combined2database_label:
            combined2database_label[str(cluster_label)] = str(len(exemplars_data))
        exemplars_data[str(cluster_label)] = exemplar_embedding.tolist()


    # Store exemplar images so it could easily be named by user
    for image_name, cluster_label in zip(new_exemplars_names, new_combined_cluster):
        image_path = os.path.join(INPUT_DIRECTORY, image_name)
        extension = image_name.split(".")[-1]
        destination = os.path.join(EXEMPLARS_DIRECTORY, combined2database_label[cluster_label] + "." + extension)

        try:
            shutil.copyfile(image_path, destination)
        except WindowsError:
            os.remove(destination)
            shutil.copyfile(image_path, destination)

    # Update the exemplar database file
    with open(EXEMPLARS_DATABASE, "w") as f:
        json.dump(exemplars_data, f)


    # Transfer files to respective cluster names
    with open(NAME_MAP_DATABASE, "r") as f:
        cluster_name = json.load(f)

    # Map to connect new_cluster_labels to combined_cluster_labels
    new2combined_label = dict(zip(map(str, new_cluster_labels[new_exemplars]), new_combined_cluster))
    for image_name, cluster in zip(image_names, new_cluster_labels):
        image_path = os.path.join(INPUT_DIRECTORY, image_name)

        # Cluster this image belongs in database
        proposed_cluster_name = combined2database_label[new2combined_label[str(cluster)]]
        destination_directory = os.path.join(OUTPUT_DIRECTORY, cluster_name.get(proposed_cluster_name, proposed_cluster_name))
        destination = os.path.join(destination_directory, image_name)
        if not os.path.exists(destination_directory):
            os.mkdir(destination_directory)

        try:
            os.rename(image_path, destination)
        except WindowsError:
            os.remove(destination)
            os.rename(image_path, destination)


if __name__ == "__main__":
    main()



import numpy as np
from pathlib import Path
import torch
from sklearn.cluster import KMeans
import argparse

#embeddings_dir = "/home/ubuntu/data/processed_video/img_embeddings_black"
embeddings_dir = "/home/ubuntu/data/processed_video/phq9_multiclass_keypoints"
class_dict = {"minimal": 0, "mildLow": 1, "modMedium": 2, "severeHigh": 3}
embed_length = 512

def get_embeddings(src_root_path, zoom_only=False):
    embeddings_labeled = []
    embeddings_only = []

    for class_folder in sorted(src_root_path.iterdir()):
        class_name = str(class_folder).split("/")[-1]
        if class_name == "mh_01.json":
            continue
        class_num = class_dict[class_name]
        for video_folder in sorted(class_folder.iterdir()):
            video_name = str(video_folder).split("/")[-1]
            if (zoom_only and video_name.split("_")[0] == "inperson"):
                continue
            for image_file in sorted(video_folder.iterdir()):
                image_name = str(image_file).split("/")[-1]
                full_path = str(src_root_path) + "/" + str(class_name) + "/" + str(video_name) + "/" + str(image_name)
                img_embedding = torch.load(full_path).detach().numpy()
                embeddings_labeled.append((class_num, video_name, image_name, img_embedding))
                embeddings_only.append(img_embedding)
        print("Got embeddings for class " + str(class_name))
    print("Got all embeddings\n")

    return embeddings_labeled, embeddings_only

def do_kmeans(embeddings, num_clusters):
    # Do k-means clustering
    kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
    print("Finished doing kmeans")
    centroids = kmeans.cluster_centers_
    print("Cluster centers = " + str(centroids))
    dist = np.linalg.norm(np.array(centroids[0]) - np.array(centroids[1]))
    print("distance between clusters 0 and 1 = " + str(dist) + "\n")
    return kmeans

def predict_video(class_name, video_name, kmeans):
    num_classes = len(kmeans.cluster_centers_)
    video_embeddings = []
    video_folder = Path(embeddings_dir) / class_name / video_name 
    for image_file in sorted(video_folder.iterdir()):
        image_name = str(image_file).split("/")[-1]
        img_embedding = torch.load(image_file).detach().numpy()[0]
        video_embeddings.append(img_embedding)
    result = kmeans.predict(video_embeddings)
    class_counts = [0] * num_classes
    for i in range(num_classes):
        class_counts[result[i]] += 1
    best_class = np.argmax(np.array(class_counts))
    print("Video: " + str(video_name) + " -- kmeans predicted cluster: " + str(best_class))
    return class_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'num_clusters', default=2, type=int, help="number of clusters for kmeans")
    parser.add_argument(
        'src_dir_path', default=None, type=Path, help='Directory path of img embeddings')
    parser.add_argument(
        'zoom_only', default=False, type=bool, help='Only use zoom videos')
    args = parser.parse_args()

    if (args.src_dir_path is not None):
        embeddings_dir = args.src_dir_path
    
    embeddings_labeled, embeddings_only = get_embeddings(embeddings_dir, args.zoom_only)
    kmeans = do_kmeans(embeddings_only, args.num_clusters)

    # Try 2 videos from each actual class
    # minimal
    predict_video("minimal", "inperson_14", kmeans)
    predict_video("minimal", "inperson_15", kmeans)
    predict_video("minimal", "zoom_100", kmeans)
    predict_video("minimal", "zoom_102", kmeans)
    predict_video("minimal", "zoom_104", kmeans)
    print("\n")
    # mildLow
    predict_video("mildLow", "inperson_13", kmeans)
    predict_video("mildLow", "inperson_16", kmeans)
    predict_video("mildLow", "zoom_35", kmeans)
    predict_video("mildLow", "zoom_37", kmeans)
    predict_video("mildLow", "zoom_38", kmeans)
    print("\n")
    # modMedium
    predict_video("modMedium", "inperson_10", kmeans)
    predict_video("modMedium", "inperson_23", kmeans)
    predict_video("modMedium", "zoom_34", kmeans)
    predict_video("modMedium", "zoom_61", kmeans)
    predict_video("modMedium", "zoom_73", kmeans)
    print("\n")
    # severeHigh
    predict_video("severeHigh", "inperson_1", kmeans)
    predict_video("severeHigh", "inperson_8", kmeans)
    predict_video("severeHigh", "zoom_48", kmeans)
    predict_video("severeHigh", "zoom_58", kmeans)
    predict_video("severeHigh", "zoom_67", kmeans)
    print("\n")

    print("Done")

main()

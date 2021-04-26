import subprocess
import argparse
from pathlib import Path
from PIL import Image

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import csv

from joblib import Parallel, delayed

from subprocess import PIPE #added by Ruta

image_name_formatter=lambda x: f'image_{x:05d}.jpg'

resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embeddings2(src_root_path, dst_root_path):
    # Go through each image in the src_root_path
    # Get an embedding using facenet
    # Save embedding vectors to a file in dst_root_path
    for class_folder in sorted(src_root_path.iterdir()):
        class_name = str(class_folder).split("/")[-1]
        if (class_name == "mh_01.json"):
            continue
        os.mkdir(dst_root_path / class_name)
        for video_folder in sorted(class_folder.iterdir()):
            video_name = str(video_folder).split("/")[-1]
            os.mkdir(dst_root_path / class_name / video_name)
            for image_file in sorted(video_folder.iterdir()):
                image_name = str(image_file).split("/")[-1]
                full_img_path = src_root_path / class_name / video_name / image_name
                img = Image.open(full_img_path)
                img_tensor = torch.from_numpy(np.asarray(img))
                img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
                img_embedding = resnet(img_tensor)
                outfile = str(dst_root_path / class_name / video_name / image_name.split(".")[0]) + ".pt"
                #with open(outfile, mode="w+") as embedfile:
                    #embedfile.write(str(img_embedding))
                torch.save(img_embedding, outfile)
        print("Done with class: " + str(class_name))


def get_embeddings(dst_root_path, outfile):
    # Go through each image in the dst_root_path
    # Get an embedding using facenet
    # Save embedding vectors to a csv
    with open(outfile, mode='w+') as csvfile: 
        embed_writer = csv.writer(csvfile, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for class_folder in sorted(dst_root_path.iterdir()):
            class_name = str(class_folder).split("/")[-1]
            if (class_name == "mh_01.json"):
                continue
            for video_folder in sorted(class_folder.iterdir()):
                video_name = str(video_folder).split("/")[-1]
                for image_file in sorted(video_folder.iterdir()):
                    image_name = str(image_file).split("/")[-1]
                    full_img_path = dst_root_path / class_name / video_name / image_name
                    img = Image.open(full_img_path)
                    img_tensor = torch.from_numpy(np.asarray(img))
                    img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
                    img_embedding = resnet(img_tensor)
                    row = [image_name, video_name, class_name, img_embedding]
                    embed_writer.writerow(row)
            print("Done with class: " + str(class_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path', default=None, type=Path, help='Directory path of img embeddings')
    #parser.add_argument(
    #    'csv_path', default='img_embeddings.csv', type=Path, help='Path to output csv')
    args = parser.parse_args()
    
    #print("Getting csv: " + str(args.csv_path) + " from imgs in " + str(args.dir_path))
    get_embeddings2(args.dir_path, args.dst_path)
    print("Done")


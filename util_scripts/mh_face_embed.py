import subprocess
import argparse
from pathlib import Path
from PIL import Image, ImageOps

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import csv
import cv2
from imutils import face_utils
import dlib
import eos
from joblib import Parallel, delayed
from subprocess import PIPE #added by Ruta

image_name_formatter=lambda x: f'image_{x:05d}.jpg'
#resnet = InceptionResnetV1(pretrained='vggface2').eval()

model = eos.morphablemodel.load_model("/home/ubuntu/ruta/3d-facial-landmark-detection-and-tracking/share/sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes("/home/ubuntu/ruta/3d-facial-landmark-detection-and-tracking/share/expression_blendshapes_3448.bin")
morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
landmark_mapper = eos.core.LandmarkMapper('/home/ubuntu/ruta/3d-facial-landmark-detection-and-tracking/share/ibug_to_sfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology('/home/ubuntu/ruta/3d-facial-landmark-detection-and-tracking/share/sfm_3448_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load('/home/ubuntu/ruta/3d-facial-landmark-detection-and-tracking/share/ibug_to_sfm.txt')
model_contour = eos.fitting.ModelContour.load('/home/ubuntu/ruta/3d-facial-landmark-detection-and-tracking/share/sfm_model_contours.json')
landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings

def landmarks_3d_fitting(landmarks,image_height, image_width):
    eos_landmarks = []
    for idx in range(0,68):
        eos_landmarks.append(eos.core.Landmark(str(idx+1), [float(landmarks[idx,0]), float(landmarks[idx,1])]))
    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
        eos_landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour)

    vertices = np.array(mesh.vertices)
    vertices = np.append(vertices, np.ones((vertices.shape[0], 1)), 1)

    w2, h2 = image_width/2, image_height/2
    viewport = np.array([[w2, 0, 0, w2],
                        [0, h2*(-1), 0, h2],
                        [0, 0, 0.5, 0.5],
                        [0, 0, 0, 1]])
    a = multiplyABC(viewport, pose.get_projection() ,pose.get_modelview())
    a = a.transpose()
    mesh_3d_points = np.dot(vertices, a)

    # landmark index in mesh
    Ind = np.zeros((68,))
    for (i, (x, y)) in enumerate(landmarks):
        Ind[i] = np.argmin((np.square(x - mesh_3d_points[:, 0]) + np.square(y - mesh_3d_points[:,1])))
    rotation_angle = pose.get_rotation_euler_angles()
    return (vertices, mesh_3d_points, Ind, rotation_angle)

def multiplyABC(A, B, C):
    temp = np.dot(A, B)
    return np.dot(temp, C)

def nose_center_normalize(keypoints, width, height):
    # Center of the nose is keypoint 34
    nose = keypoints[34]
    # Get vector from center of image to center of nose
    img_center = np.array([width/2, height/2])
    nose_shift = img_center - nose

    # For each keypoint, add that vector
    new_keypoints = np.copy(keypoints) + nose_shift
    # Keep track of distance to nose for each point
    distances = np.linalg.norm(new_keypoints - img_center, axis=1)
    max_distance = np.max(distances)
    # Divide each coordinate by max distance
    new_keypoints = new_keypoints / max_distance
    return new_keypoints

def get_dlib_keypoint_features_3D(src_root_path, dst_root_path):
    # Get the face detector and predictor from dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Go through each image in the src_root_path
    # Get an embedding using dlib facial keypoint features
    # Save embedding to a file in dst_root_path
    for class_folder in sorted(src_root_path.iterdir()):
        class_name = str(class_folder).split("/")[-1]
        if (class_name == "mh_01.json"):
            continue
        print("Making a classdir for class: " + str(class_name))
        os.mkdir(dst_root_path / class_name)
        for video_folder in sorted(class_folder.iterdir()):
            video_name = str(video_folder).split("/")[-1]
            os.mkdir(dst_root_path / class_name / video_name)
            for image_file in sorted(video_folder.iterdir()):
                image_name = str(image_file).split("/")[-1]
                full_img_path = src_root_path / class_name / video_name / image_name

                #img = Image.open(full_img_path)
                img = cv2.imread(str(full_img_path))
                #gray_img = ImageOps.grayscale(img)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_img, 1)
                #print("faces = " + str(faces))
                height, width = gray_img.shape
                face = dlib.rectangle(left=0, top=0, right=width-1, bottom=height-1)
                if (len(faces) > 0):
                    face = faces[0]
                keypoints = predictor(gray_img, face)
                keypoints = face_utils.shape_to_np(keypoints)

                # Convert 2d keypoints to 3d keypoints
                (vertices, mesh_plotting, Ind, rotation_angle) = landmarks_3d_fitting(keypoints, height, width)
                frame_landmarks_3d = vertices[np.int32(Ind),0:3]
                
                # Center the keypoints at the nose, then normalize
                #keypoints = nose_center_normalize(keypoints, width, height)

                #print("keypoints type = " + str(type(keypoints)))
                img_embedding = torch.from_numpy(frame_landmarks_3d)
                
                outfile = str(dst_root_path / class_name / video_name / image_name.split(".")[0]) + ".pt"
                #with open(outfile, mode="w+") as embedfile:
                    #embedfile.write(str(img_embedding))
                torch.save(img_embedding, outfile)
        print("Done with class: " + str(class_name))

def get_dlib_keypoint_features(src_root_path, dst_root_path):
    # Get the face detector and predictor from dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Go through each image in the src_root_path
    # Get an embedding using dlib facial keypoint features
    # Save embedding to a file in dst_root_path
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

                #img = Image.open(full_img_path)
                img = cv2.imread(str(full_img_path))
                #gray_img = ImageOps.grayscale(img)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_img, 1)
                #print("faces = " + str(faces))
                height, width = gray_img.shape
                face = dlib.rectangle(left=0, top=0, right=width-1, bottom=height-1)
                if (len(faces) > 0):
                    face = faces[0]
                keypoints = predictor(gray_img, face)
                keypoints = face_utils.shape_to_np(keypoints)
                
                # Center the keypoints at the nose, then normalize
                keypoints = nose_center_normalize(keypoints, width, height)

                #print("keypoints type = " + str(type(keypoints)))
                img_embedding = torch.from_numpy(keypoints)
                
                outfile = str(dst_root_path / class_name / video_name / image_name.split(".")[0]) + ".pt"
                #with open(outfile, mode="w+") as embedfile:
                    #embedfile.write(str(img_embedding))
                torch.save(img_embedding, outfile)
        print("Done with class: " + str(class_name))

def get_facenet_embeddings2(src_root_path, dst_root_path):
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


def get_facenet_embeddings(dst_root_path, outfile):
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
    get_dlib_keypoint_features_3D(args.dir_path, args.dst_path)
    print("Done")


import io

import h5py
from PIL import Image

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

resnet = InceptionResnetV1(pretrained='vggface2').eval()

class ImageEmbeddingsLoader(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        pieces = str(path).split("/")
        classname, videoname, filename = pieces[-3], pieces[-2], pieces[-1].split(".")[0]+".pt"
        path = "/home/ubuntu/data/processed_video/keypoints_binary_nose/" + classname + "/" + videoname + "/" + filename
        #path = "/home/ubuntu/data/processed_video/img_embeddings_binary/" + classname + "/" + videoname + "/" + filename
        img_embedding = torch.load(path).detach()
        img_embedding = torch.reshape(img_embedding, (1, 136)).type(torch.FloatTensor)
        #print(img_embedding[0, 0:10])
        #img_embedding = torch.flatten(img_embedding)
        #print("img embedding shape = " + str(img_embedding.size()))
        return img_embedding
        #with path.open('rb') as f:
            #with Image.open(f) as img:
            #    img = img.convert('RGB')
            #    img_tensor = torch.from_numpy(np.copy(np.asarray(img)))
            #    img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
            #    img_embedding = resnet(img_tensor)
            #    return img_embedding

class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path))

class EmbeddingLoader(object):
    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            #print("image path = " + str(image_path))
            if image_path.exists():
                img = self.image_loader(image_path)
                img_tensor = torch.from_numpy(np.asarray(img))
                img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
                img_embedding = resnet(img_tensor)
                video.append(img_embedding)

        return video

class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            #print("image path = " + str(image_path))
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video


class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:

            flow_data = []
            for flow in self.flows:
                flow_data.append(f[f'video_{flow}'])

            video = []
            for i in frame_indices:
                if i < len(flow_data[0]):
                    frame = [
                        Image.open(io.BytesIO(video_data[i]))
                        for video_data in flow_data
                    ]
                    frame.append(frame[-1])  # add dummy data into third channel
                    video.append(Image.merge('RGB', frame))

        return video

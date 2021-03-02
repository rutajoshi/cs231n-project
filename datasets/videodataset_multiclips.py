import json
import copy
import functools

import torch
from torch.utils.data.dataloader import default_collate

from .videodataset import VideoDataset

# START OF RUTA'S CUSTOM COLLATE
# Custom collate function for sizing issues in val data (Ruta)
import numpy as np
def custom_collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)
    
    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    maxshape = np.array(batch_clips[0]).shape
    #print("maxshape = " + str(maxshape))
    padded_batch_clips = []

    for x in batch_clips:
        x = np.array(x)
        if (x.shape != maxshape):
            #print("x.shape is " + str(x.shape) + " is not the same as " + str(maxshape))
            if (x.shape[1] < maxshape[1]):
            	x = np.pad(x, [(0,0), (0,1), (0,0), (0,0)], mode='constant', constant_values=(0))
            elif (x.shape[1] > maxshape[1]):
                x = x[::, :maxshape[1], ::, ::]
            #print("fixed so that x.shape is now = " + str(x.shape))
        padded_batch_clips.append(x)
            
    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(padded_batch_clips), default_collate(batch_targets)
    else:
        return default_collate(padded_batch_clips), batch_targets

# END OF RUTA'S CUSTOM COLLATE

def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]['video']

        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if 'segment' in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index('segment')
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]

        return clips, targets

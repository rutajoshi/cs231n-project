import subprocess
import argparse
from pathlib import Path
import json
import os

from joblib import Parallel, delayed

from subprocess import PIPE #added by Ruta
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

def cut_video(video_file_path, dst_root_path, annotations_path):
    # For the given video, pull the annotations from the annotation directory
    # Go through the json and cut the video into 11 clips
    # Store each clip in the dst_root_path directory under a folder named after the video
    
    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)

    if ("q11.mp4" in os.listdir(dst_dir_path)):
        return
    #if (len(os.listdir(dst_dir_path)) >= 11):
    #    # This video is done already
    #    return

    video_num = int(name.split("_")[1])
    if (video_num == 2):
        return # Video 2 has no audio, so we won't use it.
    annotations_file = "audio_only_" + str(video_num) + ".json"
    if (video_num < 34):
        annotations_file = "inperson_" + str(video_num) + ".json"

    if (video_num in [103, 104, 42, 60, 65, 74, 79, 78]):
        annotations_file = "audio_only_" + str(video_num) + "_fixed.json"
    print("Using annotations file: " + str(annotations_file) + " for video " + str(name))
    f = open(str(annotations_path) + "/" + annotations_file,)
    annotations = json.load(f)

    #if (len(annotations) < 22):
    #    print("Only " + str(len(annotations)) + " objects in json file: " + str(annotations_file))
    #    return

    question_num = -1
    i = 0
    while (question_num != 11):
        question = annotations[i*2]
        answer = annotations[i*2 + 1]
        assert(question["data"]["who"] == "interviewer")
        assert(answer["data"]["who"] == "participant")

        if ("info" in answer["data"] and answer["data"]["info"] == "nonanswer"):
            i += 1
            continue

        start_time = question["start"]
        end_time = answer["end"]
        question_num = answer["data"]["question"]
        question_name = "q" + str(question_num) + ".mp4"
        print("start = " + str(start_time) + "\nend = " + str(end_time))
        print(question_name + " should be " + str(end_time - start_time) + " seconds long")
        question_file_path = dst_dir_path / question_name
        #ffmpeg_extract_subclip(video_file_path, start_time, end_time, targetname=question_file_path)

        with VideoFileClip(str(video_file_path)) as video:
            clip = video.subclip(start_time, end_time)
            clip.write_videofile(str(question_file_path), audio_codec='aac')

        if (question_num == 9 and video_num in [59, 78]):
            break

        i += 1

    f.close()
    print("Finished cutting video: " + str(name))
        
def class_process(class_dir_path, dst_root_path, annotations_path):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        cut_video(video_file_path, dst_class_path, annotations_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')

    parser.add_argument(
        'ann_path',
        default=None,
        type=Path,
        help='Directory path of json annotations for cutting')

    parser.add_argument(
        'dataset',
        default='',
        type=str,
        help='Dataset name (mh | ntu | kinetics | mit | ucf101 | hmdb51 | activitynet)')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=240, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset in ['mh', 'kinetics', 'mit', 'activitynet']:
        ext = '.mp4'
    else:
        ext = '.avi'

    print("Hi there")

    if args.dataset == 'activitynet':
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(video_process)(
                video_file_path, args.dst_path, ext, args.fps, args.size)
                                 for video_file_path in video_file_paths)
    else:
        class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
        test_set_video_path = args.dir_path / 'test'
        if test_set_video_path.exists():
            class_dir_paths.append(test_set_video_path)

        args.n_jobs = 1
        print("Starting parallel jobs on: " + str(args.n_jobs) + " threads")

        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(class_process)(
                class_dir_path, args.dst_path, args.ann_path)
                                 for class_dir_path in class_dir_paths)

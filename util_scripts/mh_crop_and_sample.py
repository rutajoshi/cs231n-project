import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed

from subprocess import PIPE #added by Ruta

def crop_and_sample(video_file_path, dst_root_path, keepframes=64):
    # For every question folder, figure out how many frames there are
    # If less than 64 frames, pad to 64
    # If more than 64 frames, divide num_frames / 64 and sample until you have 64
    # For each frame in the loop that you will save, run face detection on it and crop
    # Save the cropped sampled frame to the dst_root_path / video_name folder
    print("y")

    for question_folder in sorted(video_file_path.iterdir()):
        question_folder_path = video_file_path + "/" + question_folder
        num_frames = len(question_folder_path.iterdir())
        if (num_frames < keepframes):
            # Duplicate the last frame until you've copied 64 frames to the target directory
            print("y")
        else:
            # Copy every nth file (n = num_frames / keepframes) to the target directory
            print("x")

def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append(str(video_file_path))

    #p = subprocess.run(ffprobe_cmd, capture_output=True)
    p = subprocess.run(ffprobe_cmd, stdout=PIPE, stderr=PIPE) #added by Ruta

    res = p.stdout.decode('utf-8').splitlines()
    if len(res) < 4:
        return

    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    duration = float(res[3])
    n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    # For each question, make a folder and process the video
    for video_file_path in sorted(class_dir_path.iterdir()):
        # Make a directory for the questions of this video set
        video_dst_dir_path = dst_class_path / video_file_path.stem
        video_dst_dir_path.mkdir(exist_ok=True)
        # Go through all 11 questions and process their videos
        for question_file_path in sorted(video_file_path.iterdir()):
            video_process(question_file_path, video_dst_dir_path, ext, fps, size)

    #for video_file_path in sorted(class_dir_path.iterdir()):
    #    video_process(video_file_path, dst_class_path, ext, fps, size)


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
                class_dir_path, args.dst_path, ext, args.fps, args.size)
                                 for class_dir_path in class_dir_paths)

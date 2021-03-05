import subprocess
import argparse
from pathlib import Path
from PIL import Image

from joblib import Parallel, delayed

from subprocess import PIPE #added by Ruta

def crop_and_sample(video_file_path, dst_root_path, keepframes=64):
    # For every question folder, figure out how many frames there are
    # If less than 64 frames, pad to 64
    # If more than 64 frames, divide num_frames / 64 and sample until you have 64
    # For each frame in the loop that you will save, run face detection on it and crop
    # Save the cropped sampled frame to the dst_root_path / video_name folder

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)   

    ctr = 0 # For each video, we start from 0 and count upwards through all 11 questions
 
    for question_folder in sorted(video_file_path.iterdir()):
        question_folder_path = video_file_path + "/" + question_folder
        all_frames = sorted(question_folder_path.iterdir())
        num_frames = len(all_frames)
        if (num_frames < keepframes):
            # Duplicate the last frame until you've copied 64 frames to the target directory
            for frame_file in all_frames:
                frame = Image.open(question_folder_path + "/" + frame_file) #LOAD THE IMAGE FRAME
                face_locations = face_recognition.face_locations(frame) #FIND ALL FACES
                if (len(face_locations) == 0):
                   continue
                top, right, bottom, left = face_locations[0]

                cropped_frame = frame.crop((left, top, right, bottom)) #CROP THE FRAME USING ONLY FIRST FACE
                cropped_frame.save(dst_dir_path + "/" + "img" + str(ctr) + ".jpg") #STORE THE FRAME as img + ctr + .jpg
                ctr += 1

            last_frame = all_frames[-1] #CROP THE LAST THING IN all_frames
            last_frame_locations = face_recognition.face_locations(last_frame)
            cropped_last_frame = last_frame
            if (len(last_frame_locations) != 0):
                top, right, bottom, left = last_frame_locations[0]
                cropped_last_frame = last_frame.crop((top, right, bottom, left))
            
            for i in range(keepframes - num_frames):
                cropped_last_frame.save(dst_dir_path + "/" + "img" + str(ctr) + ".jpg") #STORE THE last_frame
                ctr += 1
             
        else:
            # Copy every nth file (n = num_frames / keepframes) to the target directory
            jump = num_frames // keepframes
            for i in range(0, num_frames, jump):
                frame = Image.open(question_folder_path + "/" + frame_file) #LOAD THE IMAGE FRAME
                face_locations = face_recognition.face_locations(frame) #FIND ALL FACES
                if (len(face_locations) == 0):
                   continue
                top, right, bottom, left = face_locations[0]

                cropped_frame = frame.crop((left, top, right, bottom)) #CROP THE FRAME USING ONLY FIRST FACE
                cropped_frame.save(dst_img_path + "/" + "img" + str(ctr) + ".jpg") #STORE THE FRAME as img + ctr + .jpg
                ctr += 1

def class_jpg_process(class_dir_path, dst_root_path):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    # For each question, make a folder and process the video
    for video_file_path in sorted(class_dir_path.iterdir()):
        crop_and_sample(video_file_path, dst_class_path, keepframes=64)

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
            backend='threading')(delayed(class_jpg_process)(
                class_dir_path, args.dst_path, ext, args.fps, args.size)
                                 for class_dir_path in class_dir_paths)

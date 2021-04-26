import json
import argparse
from pathlib import Path


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def load_ground_truth(ground_truth_path, subset):
    with ground_truth_path.open('r') as f:
        data = json.load(f)

    class_labels_map = get_class_labels(data)

    # Added by RUTA
    if (subset == "train"):
        subset = "training"
    elif (subset == "val"):
        subset = "validation"

    ground_truth = []
    for video_id, v in data['database'].items():
        if subset != v['subset']:
            continue
        this_label = v['annotations']['label']
        ground_truth.append((video_id, class_labels_map[this_label]))

    return ground_truth, class_labels_map


def load_result(result_path, top_k, class_labels_map):
    with result_path.open('r') as f:
        data = json.load(f)

    result = {}
    for video_id, v in data['results'].items():
        labels_and_scores = []
        for this_result in v:
            label = class_labels_map[this_result['label']]
            score = this_result['score']
            labels_and_scores.append((label, score))
        labels_and_scores.sort(key=lambda x: x[1], reverse=True)
        result[video_id] = list(zip(*labels_and_scores[:top_k]))[0]
    return result


def remove_nonexistent_ground_truth(ground_truth, result):
    exist_ground_truth = [line for line in ground_truth if line[0] in result]

    return exist_ground_truth


def evaluate(ground_truth_path, result_path, subset, top_k, ignore):
    print('load ground truth')
    ground_truth, class_labels_map = load_ground_truth(ground_truth_path,
                                                       subset)
    print('number of ground truth: {}'.format(len(ground_truth)))

    print('load result')
    result = load_result(result_path, top_k, class_labels_map)
    print('number of result: {}'.format(len(result)))

    n_ground_truth = len(ground_truth)
    ground_truth = remove_nonexistent_ground_truth(ground_truth, result)
    if ignore:
        n_ground_truth = len(ground_truth)

    print('calculate top-{} accuracy'.format(top_k))
    correct = [1 if line[1] in result[line[0]] else 0 for line in ground_truth]
    accuracy = sum(correct) / n_ground_truth

    # Class-wise accuracy
    #gt_class_counts, re_class_counts = [0 for i in range(4)], [0 for i in range(4)]
    gt_class_counts, re_class_counts = [0 for i in range(2)], [0 for i in range(2)]
    for line in ground_truth:
        correct = 1 if line[1] in result[line[0]] else 0
        if (correct == 1):
            re_class_counts[line[1]] += 1
        gt_class_counts[line[1]] += 1
    #accuracies = [re_class_counts[i] for i in range(4)]
    accuracies = [re_class_counts[i] for i in range(2)]
    for i in range(len(gt_class_counts)):
        if (gt_class_counts[i] != 0):
            accuracies[i] = accuracies[i] / gt_class_counts[i]
    print("Class counts GT = " + str(gt_class_counts))
    print("Class counts RE = " + str(re_class_counts))
    print("Class-wise accuracies = " + str(accuracies))

    # Weighted accuracy
    #weights = [0.4022, 0.9737, 3.7000, 4.6250] # 4 classes
    weights = [0.8043, 1.3214] # 2 classes
    total_weights = sum(weights)
    #weighted_acc = sum([weights[i]*accuracies[i] for i in range(4)])
    weighted_acc = sum([weights[i]*accuracies[i] / total_weights for i in range(2)])
    print("Weighted accuracy = " + str(weighted_acc))

    print('top-{} accuracy: {}'.format(top_k, accuracy))
    print("\n")
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth_path', type=Path)
    parser.add_argument('result_path', type=Path)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--save', action='store_true')
    parser.add_argument(
        '--ignore',
        action='store_true',
        help='ignore nonexistent videos in result')

    args = parser.parse_args()

    accuracy = evaluate(args.ground_truth_path, args.result_path, args.subset,
                        args.k, args.ignore)

    if args.save:
        with (args.result_path.parent / 'top{}.txt'.format(
                args.k)).open('w') as f:
            f.write(str(accuracy))

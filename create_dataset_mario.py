import random
from itertools import product
from operator import indexOf
from random import shuffle
from PIL import Image


import torch
import os
import shutil

from tqdm import tqdm

import random


def move_dataset_to_right_folder():

    source_folder_name = "observations"
    target_folder_name = "Observations"

    unique_labels = set()

    counter = 0
    # for each folder

    # find the unique labels
    for folder in os.listdir(source_folder_name):
        for episode in os.listdir(os.path.join(source_folder_name, folder)):

            for file in os.listdir(os.path.join(source_folder_name, folder, episode)):

                label = file.split("_")[1].split(".")[0]

                unique_labels.add(int(label))

    # sort the labels and create a mapping
    unique_labels = sorted(unique_labels)
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    print(label_mapping)

    counter = 0

    for folder in os.listdir(source_folder_name):
        for episode in os.listdir(os.path.join(source_folder_name, folder)):
            # move the content of the episode folder to the target folder
            episode_folder = os.path.join(target_folder_name, f"episode_{counter}")
            if not os.path.exists(episode_folder):
                os.makedirs(episode_folder)

            for file in os.listdir(os.path.join(source_folder_name, folder, episode)):

                frame_number = file.split("_")[0]

                label = file.split("_")[1].split(".")[0]

                label = int(label)

                label = label_mapping[label]

                # put the file in the episode folder and rename it
                source_file_path = os.path.join(
                    source_folder_name, folder, episode, file
                )

                target_file_path = os.path.join(
                    episode_folder, f"{frame_number}_{label}.png"
                )

                # create a copy of the file in the target folder
                shutil.copy(source_file_path, target_file_path)

        counter += 1

    print(label_mapping)
    print("Number of unique labels:", len(unique_labels))


import os
import random

import os
import random
import math

def balance_labels():
    directory = "./Observations/"
    label_episode_files = {}  # {label: {episode: [file_paths]}}

    # Step 1: Collect files grouped by label and episode
    for episode in os.listdir(directory):
        episode_path = os.path.join(directory, episode)
        if not os.path.isdir(episode_path):
            continue

        for file in os.listdir(episode_path):
            if "_" not in file or "." not in file:
                continue
            try:
                label = int(file.split("_")[1].split(".")[0])
            except ValueError:
                continue

            label_episode_files.setdefault(label, {}).setdefault(episode, []).append(
                os.path.join(episode_path, file)
            )

    # Step 2: Calculate label totals
    label_totals = {label: sum(len(files) for files in episodes.values())
                    for label, episodes in label_episode_files.items()}
    print("Original label distribution:", label_totals)

    # Step 3: Get minimum count for balancing
    min_count = min(label_totals.values())
    print("Balancing to minimum count per label:", min_count)

    # Step 4: Proportional deletions across episodes
    for label, episodes in label_episode_files.items():
        total = label_totals[label]
        excess = total - min_count
        if excess <= 0:
            continue  # Already balanced

        print(f"\nBalancing label {label}, need to delete {excess} samples")

        # Compute how many to delete from each episode
        deletions = {}
        for episode, files in episodes.items():
            proportion = len(files) / total
            deletions[episode] = math.floor(proportion * excess)

        # Adjust for rounding issues (ensure total deletions == excess)
        to_adjust = excess - sum(deletions.values())
        if to_adjust > 0:
            # Distribute remaining deletions randomly to episodes with highest count
            sorted_eps = sorted(episodes.items(), key=lambda x: len(x[1]), reverse=True)
            for i in range(to_adjust):
                ep = sorted_eps[i % len(sorted_eps)][0]
                deletions[ep] += 1

        # Step 5: Perform deletions
        for episode, count in deletions.items():
            files = episodes[episode]
            if count > 0:
                random.shuffle(files)
                for f in files[:count]:
                    os.remove(f)
                    print(f"Deleted {f}")

    print("\nBalancing complete.")





def loadMarioDataset_balanced_labels(root_dir, num_outputs=5, transform=None):
    """
    Args:
        root_dir (str): Path to the dataset directory.
        transform (callable, optional): A function/transform to apply to the images.
    """
    image_sequences = []
    labels = []

    # Load all sequences and their image-label pairs
    for sequence_dir in tqdm(sorted(os.listdir(root_dir)), desc="Processing Sequences"):
        sequence_path = os.path.join(root_dir, sequence_dir)
        if os.path.isdir(sequence_path):
            images = []
            files = os.listdir(sequence_path)
            for filename in tqdm(files, desc=f"Processing {sequence_dir}", leave=False):
                if filename.endswith(("png", "jpg", "jpeg")):
                    label = int(
                        filename.split("_")[1].split(".")[0]
                    ) # Extract label from filename
                    """
                    if num_outputs == 3:
                        # distinction in failure, neutral, success
                        # success (->2)
                        if label == 0:
                            label = 2
                        # failure (->0)
                        elif label == 4:
                            label = 0
                        else:
                            label = 1
                    elif num_outputs == 2:
                        # distinction in failure, success
                        # success (->1)
                        if label == 0:
                            label = 1
                            # failure (->0)
                        else:
                            label = 0
                    """
                    img_path = os.path.join(sequence_path, filename)
                    image = Image.open(img_path).convert("RGB")
                    images.append(transform(image))
                    image_sequences.append(torch.stack(images, dim=0))
                    labels.append(label)
                    
            """
                space_left = len(image_sequences) - 30
                for _ in range(space_left):
                    images.append(transform(image))
                    image_sequences.append(torch.stack(images, dim=0))
                    labels.append(label)
                """
    

    img_seq_train_batch = []
    acceptance_train_batch = []
    lens = []
    print(len(image_sequences))
    print(len(labels))
    for i in range(len(image_sequences)):
        trace = image_sequences[i]
        acc = labels[i]
        lenght = trace.size()[0]
        if lenght not in lens:
            lens.append(lenght)
            img_seq_train_batch.append([])
            acceptance_train_batch.append([])
        img_seq_train_batch[lens.index(lenght)].append(trace)
        acceptance_train_batch[lens.index(lenght)].append(acc)

    img_seq_train_batch_bal = []
    acceptance_train_batch_bal = []
    # balance labels
    for i, batch in enumerate(img_seq_train_batch):
        negative_traces = [
            batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 0
        ]
        neutral_traces = [
            batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 1
        ]
        positive_traces = [
            batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 2
        ]
        other_traces = [
            batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 3
        ]
        other_other_traces = [
            batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 4
        ]
        if num_outputs == 2 and not (
            len(neutral_traces)
        ):  # and len(positive_traces) and len(negative_traces) and len(other_traces)):
            continue
        if (
            num_outputs == 3
            and (not len(positive_traces))
            and (not len(negative_traces))
        ):
            continue
        # if num_outputs == 4 and not (len(negative_traces)):
        #    continue
        max_length_count = max(
            len(positive_traces),
            len(negative_traces),
            len(neutral_traces),
            len(other_traces),
            len(other_other_traces),
        )
        # max_length_count = min(len(negative_traces), len(neutral_traces))
        # negative_traces = negative_traces[:max_length_count]
        # neutral_traces = neutral_traces[:max_length_count]

        # Repeat traces if necessary to balance batches
        if len(positive_traces) and len(positive_traces) < max_length_count:
            positive_traces = (
                positive_traces * (max_length_count // len(positive_traces))
                + positive_traces[: max_length_count % len(positive_traces)]
            )

        if len(negative_traces) and len(negative_traces) < max_length_count:
            negative_traces = (
                negative_traces * (max_length_count // len(negative_traces))
                + negative_traces[: max_length_count % len(negative_traces)]
            )

        if len(neutral_traces) and len(neutral_traces) < max_length_count:
            neutral_traces = (
                neutral_traces * (max_length_count // len(neutral_traces))
                + neutral_traces[: max_length_count % len(neutral_traces)]
            )

        if len(other_traces) and len(other_traces) < max_length_count:
            other_traces = (
                other_traces * (max_length_count // len(other_traces))
                + other_traces[: max_length_count % len(other_traces)]
            )

        if len(other_other_traces) and len(other_other_traces) < max_length_count:
            other_other_traces = (
                other_other_traces * (max_length_count // len(other_other_traces))
                + other_other_traces[: max_length_count % len(other_other_traces)]
            )

        # Combine and shuffle traces
        balanced_traces = (
            positive_traces
            + negative_traces
            + neutral_traces
            + other_traces
            + other_other_traces
        )
        balanced_labels = (
            [2] * len(positive_traces)
            + [0] * len(negative_traces)
            + [1] * len(neutral_traces)
            + [3] * len(other_traces)
            + [4] * len(other_other_traces)
        )

        shuffle_indices = list(range(len(balanced_traces)))
        shuffle(shuffle_indices)

        balanced_traces = [balanced_traces[i] for i in shuffle_indices]
        balanced_labels = [balanced_labels[i] for i in shuffle_indices]

        img_seq_train_batch_bal.append(balanced_traces)
        acceptance_train_batch_bal.append(balanced_labels)

    img_seq_train_batch_bal = [torch.stack(batch) for batch in img_seq_train_batch_bal]
    acceptance_train_batch_bal = [
        torch.LongTensor(batch) for batch in acceptance_train_batch_bal
    ]

    for i in range(len(img_seq_train_batch_bal)):
        print(acceptance_train_batch_bal[i])
    return img_seq_train_batch_bal, acceptance_train_batch_bal


#############################################################################################################################################################


if __name__ == "__main__":

    move_dataset_to_right_folder()
    balance_labels()

    pass

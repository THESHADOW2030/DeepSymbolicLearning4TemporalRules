import random
from itertools import product
from operator import indexOf
from random import shuffle
from PIL import Image



import torch
import os

########################################################################################################################################
# return:    X_train, X_test, y_train, y_test
#
#            X.size = N x LengthTraces x NumSymbols
#            y.size = N x LengthTraces

# test and train traces have the same length
# output: tensor
def create_complete_set_traces_one_true_literal(length_traces, alphabet, dfa, train_size, verbose=False):
    traces = []
    traces_t = []
    accepted_list = []

    prod = product(alphabet, repeat=length_traces)

    for trace in list(prod):
            #print("trace: ", trace)
            t = []
            accepted= torch.zeros(len(trace))
            t_t = torch.zeros((len(trace), len(alphabet)))

            for step, true_literal in enumerate(trace):
                truth_v = {}
                for s, symbol in enumerate(alphabet):
                    if symbol == true_literal:
                        truth_v[symbol] = True
                        t_t[step, s] = 1.0
                    else:
                        truth_v[symbol] = False

                t.append(truth_v)
                accepted[step]= int(dfa.accepts(t))


            traces.append(t)
            traces_t.append(t_t)
            accepted_list.append(accepted)

    #shuffle
    dataset = list(zip(traces_t, accepted_list))
    random.shuffle(dataset)
    traces_t, accepted = zip(*dataset)

    #split
    split_index = round(len(traces) * train_size)

    traces_t_train = torch.stack(traces_t[:split_index])
    traces_t_test = torch.stack(traces_t[split_index:])

    accepted_train = torch.stack(accepted_list[:split_index])
    accepted_test = torch.stack(accepted_list[split_index:])

    if verbose:
        train_size = traces_t_train.size()[0]
        test_size = traces_t_test.size()[0]
        print("created symbolic dataset with all the {} traces of length {} composed by {} symbols; {} train, {} test".format(train_size+test_size, length_traces, len(alphabet), train_size, test_size))

    return traces_t_train, traces_t_test, accepted_train, accepted_test

########################################################################################################################################
# return:    X_train, X_test, y_train, y_test
#
#            X.size = N x LengthTraces x NumSymbols
#            y.size = N (only final label)

# test traces are longer length_test_traces
# train traces have length from 1 to max_length_train_traces
#output list of tensors
def create_complete_set_traces_one_true_literal_2(max_length_train_traces, length_test_traces, alphabet, dfa, verbose=False):
    dataset_X_train = []
    dataset_y_train = []

    dataset_X_test = []
    dataset_y_test = []

    positive_train = 0
    all_train = 0
    positive_test = 0
    all_test = 0

    for length_traces in list(range(1,max_length_train_traces +1)) + [length_test_traces]:
        prod = product(alphabet, repeat=length_traces)
        traces_t = []
        accepted_list = []

        for trace in list(prod):
                #print("trace: ", trace)
                t = []
                t_t = torch.zeros((len(trace), len(alphabet)))

                for step, true_literal in enumerate(trace):
                    truth_v = {}
                    for s, symbol in enumerate(alphabet):
                        if symbol == true_literal:
                            truth_v[symbol] = True
                            t_t[step, s] = 1.0
                        else:
                            truth_v[symbol] = False

                    t.append(truth_v)
                accepted= int(dfa.accepts(t))

                traces_t.append(t_t)
                accepted_list.append(accepted)

        if length_traces == length_test_traces:
            dataset_X_test.append(torch.stack(traces_t))
            dataset_y_test.append(torch.LongTensor(accepted_list))
            all_test += len(accepted_list)
            positive_test += sum(accepted_list)
        else:
            dataset_X_train.append(torch.stack(traces_t))
            dataset_y_train.append(torch.LongTensor(accepted_list))
            all_train += len(accepted_list)
            positive_train += sum(accepted_list)


    if verbose:
        for i in range(len(dataset_X_train)):
            print("train batch {}___________".format(i))
            print(dataset_X_train[i].size())
            print(dataset_y_train[i].size())
            print(dataset_X_train[i])
            print(dataset_y_train[i])
        for i in range(len(dataset_X_test)):
            print("test batch {}___________".format(i))
            print(dataset_X_test[i].size())
            print(dataset_y_test[i].size())
            print(dataset_X_test[i])
            print(dataset_y_test[i])

        #print("created symbolic dataset with all the {} traces of length {} composed by {} symbols; {} train, {} test".format(train_size+test_size, length_traces, len(alphabet), train_size, test_size))
    print("RATES OF POSITIVE TRACES")
    print("train_dataset:", positive_train / float(all_train))
    print("test_dataset:", positive_test / float(all_test))
    return dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test

def create_traces_set_one_true_literal_balanced(max_length_train_traces, length_test_traces, max_num_train_traces, alphabet, dfa, verbose=False):
    dataset_X_train = []
    dataset_y_train = []

    dataset_X_test = []
    dataset_y_test = []

    traces_t = []
    accepted_list = []
    for length_traces in list(range(1, max_length_train_traces + 1)):
        prod = list(product(alphabet, repeat=length_traces))

        for trace in prod:
            t_t = torch.zeros((len(trace), len(alphabet)))
            t = []

            for step, true_literal in enumerate(trace):
                truth_v = {}
                for s, symbol in enumerate(alphabet):
                    if symbol == true_literal:
                        truth_v[symbol] = True
                        t_t[step, s] = 1.0
                    else:
                        truth_v[symbol] = False

                t.append(truth_v)
            accepted = int(dfa.accepts(t))

            traces_t.append(t_t)
            accepted_list.append(accepted)
    # Separate positive and negative traces
    positive_traces = [traces_t[i] for i in range(len(accepted_list)) if accepted_list[i] == 1]
    negative_traces = [traces_t[i] for i in range(len(accepted_list)) if accepted_list[i] == 0]

    min_count = min(len(positive_traces), len(negative_traces))
    if max_num_train_traces <= min_count:
        min_count = max_num_train_traces
    #shuffle(positive_traces) # no shuffle because I prefer shorter traces in the training set
    #shuffle(negative_traces)

    balanced_traces = positive_traces[:min_count] + negative_traces[:min_count]
    balanced_labels = [1] * min_count + [0] * min_count

    shuffle_indices = list(range(len(balanced_traces)))
    shuffle(shuffle_indices)

    balanced_traces = [balanced_traces[i] for i in shuffle_indices]
    balanced_labels = [balanced_labels[i] for i in shuffle_indices]

    #from list of tensors to list of batches

    lens = []
    for i in range(len(balanced_traces)):
        trace = balanced_traces[i]
        acc = balanced_labels[i]
        lenght = trace.size()[0]
        if lenght not in lens:
            lens.append(lenght)
            dataset_X_train.append([])
            dataset_y_train.append([])
        dataset_X_train[lens.index(lenght)].append(trace)
        dataset_y_train[lens.index(lenght)].append(acc)

    dataset_X_train = [torch.stack(batch) for batch in dataset_X_train]
    dataset_y_train = [torch.LongTensor(batch) for batch in dataset_y_train]

    if verbose:
        for i in range(len(dataset_X_train)):
            print("train batch {}___________".format(i))
            print(dataset_X_train[i].size())
            #print(dataset_y_train[i].size())
            print("positive rate", torch.mean(dataset_y_train[i].float()))

    ####test dataset
    prod = list(product(alphabet, repeat=length_test_traces))
    traces_t = []
    accepted_list = []
    for trace in prod:
        t_t = torch.zeros((len(trace), len(alphabet)))
        t = []

        for step, true_literal in enumerate(trace):
            truth_v = {}
            for s, symbol in enumerate(alphabet):
                if symbol == true_literal:
                    truth_v[symbol] = True
                    t_t[step, s] = 1.0
                else:
                    truth_v[symbol] = False

            t.append(truth_v)
        accepted = int(dfa.accepts(t))

        traces_t.append(t_t)
        accepted_list.append(accepted)

    dataset_X_test.append(torch.stack(traces_t))
    dataset_y_test.append(torch.LongTensor(accepted_list))
    all_test = len(accepted_list)
    positive_test = sum(accepted_list)

    print("STATISTICS")
    print("train_dataset\npositive rate: {}\tnum of traces: {}".format( sum(balanced_labels) / float(len(balanced_labels)), len(balanced_traces)))
    #print(balanced_traces)
    print("test_dataset\npositive rate: {}\tnum of traces: {}".format( positive_test / float(all_test), all_test))
    return dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test


def create_traces_set_one_true_literal_balanced_batches(max_length_train_traces, length_test_traces, max_num_train_traces, alphabet, dfa, verbose=False):
    dataset_X_train = []
    dataset_y_train = []

    dataset_X_test = []
    dataset_y_test = []

    traces_t = []
    accepted_list = []
    for length_traces in list(range(1, max_length_train_traces + 1)):
        prod = list(product(alphabet, repeat=length_traces))

        for trace in prod:
            t_t = torch.zeros((len(trace), len(alphabet)))
            t = []

            for step, true_literal in enumerate(trace):
                truth_v = {}
                for s, symbol in enumerate(alphabet):
                    if symbol == true_literal:
                        truth_v[symbol] = True
                        t_t[step, s] = 1.0
                    else:
                        truth_v[symbol] = False

                t.append(truth_v)
            accepted = int(dfa.accepts(t))

            traces_t.append(t_t)
            accepted_list.append(accepted)
    # Separate positive and negative traces
    positive_traces = [traces_t[i] for i in range(len(accepted_list)) if accepted_list[i] == 1]
    negative_traces = [traces_t[i] for i in range(len(accepted_list)) if accepted_list[i] == 0]

    for length_traces in range(1, max_length_train_traces + 1):
        positive_traces_length = [trace for trace in positive_traces if trace.size(0) == length_traces]
        negative_traces_length = [trace for trace in negative_traces if trace.size(0) == length_traces]

        max_length_count = max(len(positive_traces_length), len(negative_traces_length))

        # Repeat traces if necessary to balance batches
        if len(positive_traces_length) < max_length_count and len(positive_traces_length) > 0:
            positive_traces_length = positive_traces_length * (max_length_count // len(positive_traces_length)) + \
                                     positive_traces_length[:max_length_count % len(positive_traces_length)]

        if len(negative_traces_length) < max_length_count and len(negative_traces_length) > 0:
            negative_traces_length = negative_traces_length * (max_length_count // len(negative_traces_length)) + \
                                     negative_traces_length[:max_length_count % len(negative_traces_length)]

        # Combine and shuffle traces
        balanced_traces = positive_traces_length + negative_traces_length
        balanced_labels = [1] * len(positive_traces_length) + [0] * len(negative_traces_length)

        shuffle_indices = list(range(len(balanced_traces)))
        shuffle(shuffle_indices)

        balanced_traces = [balanced_traces[i] for i in shuffle_indices]
        balanced_labels = [balanced_labels[i] for i in shuffle_indices]

        dataset_X_train.append(torch.stack(balanced_traces))

        dataset_y_train.append(torch.LongTensor(balanced_labels))

    if verbose:
        for i in range(len(dataset_X_train)):
            print("train batch {}___________".format(i))
            print(dataset_X_train[i].size())
            #print(dataset_y_train[i].size())
            print("positive rate", torch.mean(dataset_y_train[i].float()))

    ####test dataset
    prod = list(product(alphabet, repeat=length_test_traces))
    traces_t = []
    accepted_list = []
    for trace in prod:
        t_t = torch.zeros((len(trace), len(alphabet)))
        t = []

        for step, true_literal in enumerate(trace):
            truth_v = {}
            for s, symbol in enumerate(alphabet):
                if symbol == true_literal:
                    truth_v[symbol] = True
                    t_t[step, s] = 1.0
                else:
                    truth_v[symbol] = False

            t.append(truth_v)
        accepted = int(dfa.accepts(t))

        traces_t.append(t_t)
        accepted_list.append(accepted)

    dataset_X_test.append(torch.stack(traces_t))
    dataset_y_test.append(torch.LongTensor(accepted_list))
    all_test = len(accepted_list)
    positive_test = sum(accepted_list)

    print("STATISTICS")
    print("train_dataset\npositive rate: {}\tnum of traces: {}".format( sum(balanced_labels) / float(len(balanced_labels)), len(balanced_traces)))
    #print(balanced_traces)
    print("test_dataset\npositive rate: {}\tnum of traces: {}".format( positive_test / float(all_test), all_test))
    return dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test

def create_image_sequence_dataset(image_data, numb_of_classes, traces, acceptance, print_size=False):

    traces_list = []
    acceptance_list = []
    for batch in traces:
        for trace in batch:
            traces_list.append(trace)

    for batch in acceptance:
        for acc in batch:
            acceptance_list.append(acc)

    traces = traces_list
    acceptance = acceptance_list

    channels = 1
    pixels_h, pixels_v = image_data.data[0].size()
    how_many = []
    data_for_classes = []

    labels = [0,6]
    for label in labels:

        #for label in range(numb_of_classes):
        indices_i = image_data.targets == label
        data_i, target_i = image_data.data[indices_i], image_data.targets[indices_i]
        how_many.append(len(data_i))
        data_for_classes.append(data_i)

    num_of_images = sum(how_many)

    img_seq_train = []
    acceptance_train = []

    i_i = [0 for _ in range(len(how_many)) ]
    seen_images = sum(i_i)


    while True:
        for j in range(len(traces)):
            x = traces[j]
            a = acceptance[j]
            num_img = len(x)
            x_i_img = torch.zeros(num_img, channels,pixels_h, pixels_v)

            for step in range(num_img):
                if x[step][0] > 0.5:

                    x_i_img[step] = data_for_classes[0][i_i[0]]
                    i_i[0] += 1
                    if i_i[0] >= how_many[0]:
                        break
                else:
                    x_i_img[step] = data_for_classes[1][i_i[1]]
                    i_i[1] += 1
                    if i_i[1] >= how_many[1]:
                        break
            if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
                break
            img_seq_train.append(x_i_img)
            acceptance_train.append(a)

            seen_images +=num_img
        if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
            break

    img_seq_train_batch = []
    acceptance_train_batch = []
    lens = []
    for i in range(len(img_seq_train)):
        trace = img_seq_train[i]
        acc = acceptance_train[i]
        lenght = trace.size()[0]
        if lenght not in lens:
            lens.append(lenght)
            img_seq_train_batch.append([])
            acceptance_train_batch.append([])
        img_seq_train_batch[lens.index(lenght)].append(trace)
        acceptance_train_batch[lens.index(lenght)].append(acc)

    img_seq_train_batch = [torch.stack(batch) for batch in img_seq_train_batch]
    acceptance_train_batch = [torch.stack(batch) for batch in acceptance_train_batch]

    if print_size:
        print("Created image dataset with {} sequences of images".format(sum([batch.size()[0] for batch in acceptance_train_batch])))
    return img_seq_train_batch, acceptance_train_batch

def loadMinecraftDataset(root_dir, transform=None, num_outputs = 4):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (callable, optional): A function/transform to apply to the images.
        """
        image_sequences = []
        labels = []

        # Load all sequences and their image-label pairs
        for sequence_dir in sorted(os.listdir(root_dir)):
            sequence_path = os.path.join(root_dir, sequence_dir)
            if os.path.isdir(sequence_path):
                images = []
                for filename in os.listdir(sequence_path):
                    if filename.endswith(('png', 'jpg', 'jpeg')):
                        label = int(filename.split('_')[1].split('.')[0])  # Extract label from filename
                        if num_outputs == 3:
                            #distinction in failure, neutral, success
                            #success (->2)
                            if label == 0:
                                label = 2
                            #failure (->0)
                            elif label == 4:
                                label = 0
                            # neutral (->1)
                            else:
                                label = 1
                        if num_outputs == 2:
                            #distinction in failure, success
                            #success (->1)
                            if label == 0:
                                label = 1
                            #failure (->0)
                            else:
                                label = 0

                        img_path = os.path.join(sequence_path, filename)
                        image = Image.open(img_path).convert("RGB")
                        images.append(transform(image))
                        image_sequences.append(torch.stack(images, dim=0))
                        labels.append(label)
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

        img_seq_train_batch = [torch.stack(batch) for batch in img_seq_train_batch]
        acceptance_train_batch = [torch.LongTensor(batch) for batch in acceptance_train_batch]
        for i in range(len(img_seq_train_batch)):
            print(acceptance_train_batch[i])
        return  img_seq_train_batch, acceptance_train_batch


def loadMinecraftDataset_balanced_labels(root_dir, num_outputs = 5,transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (callable, optional): A function/transform to apply to the images.
        """
        image_sequences = []
        labels = []

        # Load all sequences and their image-label pairs
        for sequence_dir in sorted(os.listdir(root_dir)):
            sequence_path = os.path.join(root_dir, sequence_dir)
            if os.path.isdir(sequence_path):
                images = []
                for filename in os.listdir(sequence_path):
                    if filename.endswith(('png', 'jpg', 'jpeg')):
                        label = int(filename.split('_')[1].split('.')[0])  # Extract label from filename
                        if num_outputs == 3:
                            #distinction in failure, neutral, success
                            #success (->2)
                            if label == 0:
                                label = 2
                            #failure (->0)
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

                        img_path = os.path.join(sequence_path, filename)
                        image = Image.open(img_path).convert("RGB")
                        images.append(transform(image))
                        image_sequences.append(torch.stack(images, dim=0))
                        labels.append(label)
                '''
                space_left = len(image_sequences) - 30
                for _ in range(space_left):
                    images.append(transform(image))
                    image_sequences.append(torch.stack(images, dim=0))
                    labels.append(label)
                '''

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
        #balance labels
        for i, batch in enumerate(img_seq_train_batch):
            negative_traces = [batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 0]
            neutral_traces = [batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 1]
            positive_traces = [batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 2]
            other_traces = [batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 3]
            other_other_traces = [batch[k] for k in range(len(batch)) if acceptance_train_batch[i][k] == 4]
            if  num_outputs == 2 and not (len(neutral_traces) ):#and len(positive_traces) and len(negative_traces) and len(other_traces)):
                continue
            if num_outputs == 3 and (not len(positive_traces)) and (not len(negative_traces)):
                continue
            #if num_outputs == 4 and not (len(negative_traces)):
            #    continue
            max_length_count = max(len(positive_traces), len(negative_traces), len(neutral_traces), len(other_traces), len(other_other_traces))
            #max_length_count = min(len(negative_traces), len(neutral_traces))
            #negative_traces = negative_traces[:max_length_count]
            #neutral_traces = neutral_traces[:max_length_count]

            #Repeat traces if necessary to balance batches
            if len(positive_traces) and len(positive_traces) < max_length_count:
                positive_traces= positive_traces* (max_length_count // len(positive_traces)) + \
                                         positive_traces[:max_length_count % len(positive_traces)]

            if len(negative_traces) and len(negative_traces) < max_length_count:
                negative_traces = negative_traces * (max_length_count // len(negative_traces)) + \
                                         negative_traces[:max_length_count % len(negative_traces)]

            if len(neutral_traces) and len(neutral_traces) < max_length_count:
                neutral_traces = neutral_traces * (max_length_count // len(neutral_traces)) + \
                                        neutral_traces[:max_length_count % len(neutral_traces)]

            if len(other_traces) and len(other_traces) < max_length_count:
                other_traces = other_traces * (max_length_count // len(other_traces)) + \
                                        other_traces[:max_length_count % len(other_traces)]

            if len(other_other_traces) and len(other_other_traces) < max_length_count:
                other_other_traces = other_other_traces * (max_length_count // len(other_other_traces)) + \
                                        other_other_traces[:max_length_count % len(other_other_traces)]

            # Combine and shuffle traces
            balanced_traces = positive_traces + negative_traces + neutral_traces + other_traces + other_other_traces
            balanced_labels = [2] * len(positive_traces) + [0] * len(negative_traces) + [1] * len(
                neutral_traces) + [3] * len(other_traces) + [4]* len(other_other_traces)

            shuffle_indices = list(range(len(balanced_traces)))
            shuffle(shuffle_indices)

            balanced_traces = [balanced_traces[i] for i in shuffle_indices]
            balanced_labels = [balanced_labels[i] for i in shuffle_indices]

            img_seq_train_batch_bal.append(balanced_traces)
            acceptance_train_batch_bal.append(balanced_labels)

        img_seq_train_batch_bal = [torch.stack(batch) for batch in img_seq_train_batch_bal]
        acceptance_train_batch_bal = [torch.LongTensor(batch) for batch in acceptance_train_batch_bal]

        for i in range(len(img_seq_train_batch_bal)):
            print(acceptance_train_batch_bal[i])
        return  img_seq_train_batch_bal, acceptance_train_batch_bal
#############################################################################################################################################################

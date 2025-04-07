import sys
from copy import deepcopy

import torch
import torchvision
from pygame.transform import threshold

from DeepAutoma import ProbabilisticAutoma, LSTMClassifier, GRUClassifier, TransformerClassifier
from Classifier import CNN_mnist, CNN_minecraft, MNIST_Net
from statistics import mean
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import math
import pickle
from minimization import minimize_dfa_symbols_and_states
from utils import eval_accuracy, eval_image_classification_from_traces, gumbel_softmax, eval_accuracy_DFA
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
#import torchvision.ops.focal_loss as crossentropy

class DSLMooreMachine:
    def __init__(self, ltl_formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, num_of_symbols, num_of_states, dataset='MNIST', automa_implementation = 'logic_circuit', num_exp=0,log_dir="Results/",  automata_dir= "Automata/", models_dir= "Models"):
        self.log_dir = log_dir
        self.automata_dir = automata_dir
        self.models_dir = models_dir
        self.exp_num=num_exp
        self.ltl_formula_string = ltl_formula
        self.formula_name = formula_name
        self.dfa = dfa
        self.mutually_exclusive = True
        #save the dfa image
        try:
            self.dfa.to_graphviz().render(self.automata_dir+self.formula_name)
        except:
            print("impossible to print the target automaton")

        self.numb_of_symbols = num_of_symbols
        #self.numb_of_states = self.dfa._state_counter

        #self.alphabet = ["c"+str(i) for i in range(self.numb_of_symbols) ]
        #self.final_states = list(self.dfa._final_states)

        #reduced dfa for single label image classification
        #if self.mutually_exclusive:
        #    self.reduced_dfa = self.reduce_dfa()
        #else:
        #    self.reduced_dfa = self.reduce_dfa_non_mutex()

        #print("DFA: ",self.dfa._transition_function)

        if dataset == 'MNIST':
            self.num_channels = 1
            nodes_linear = 54

            self.pixels_h = 28
            self.pixels_v = 28
            self.classifier = CNN_mnist(self.num_channels, self.numb_of_symbols, nodes_linear)
            #self.classifier = MNIST_Net(self.numb_of_symbols, self.num_channels)

            self.num_outputs = 2


        elif dataset == 'minecraft':
            self.num_channels = 3
            self.pixels_h = 32
            self.pixels_v = 32
            nodes_linear = 96
            self.classifier = CNN_mnist(self.num_channels, self.numb_of_symbols, nodes_linear)

            self.num_outputs = 2

        else:
            sys.exit("INVALID DATASET. Choices available: ['MNIST', 'minecraft']")

        #################### networks
        self.hidden_dim =num_of_states
        self.automa_implementation = automa_implementation

        if self.automa_implementation == "logic_circuit":
            self.deepAutoma = ProbabilisticAutoma(self.numb_of_symbols, self.hidden_dim, self.num_outputs)
        elif self.automa_implementation == "lstm":
            self.deepAutoma = LSTMClassifier(self.hidden_dim, self.numb_of_symbols, self.num_outputs)
        elif self.automa_implementation == "gru":
            self.deepAutoma = GRUClassifier(self.hidden_dim, self.numb_of_symbols, self.num_outputs)
        elif self.automa_implementation == "transformer":
            self.deepAutoma = TransformerClassifier(self.hidden_dim, self.numb_of_symbols, self.num_outputs)
        else:
            sys.exit("INVALID AUTOMA IMPLEMENTATION. Choose between 'lstm', 'gru', 'transformer', and 'logic_circuit'")
        #dataset
        #self.train_traces, self.test_traces, train_acceptance_tr, test_acceptance_tr = symbolic_dataset
        self.train_img_seq, self.train_acceptance_img, self.test_img_seq_clss, self.test_acceptance_img_clss, self.test_img_seq_aut, self.test_acceptance_img_aut, self.test_img_seq_hard, self.test_acceptance_img_hard = image_seq_dataset

    def reduce_dfa(self):
        dfa = self.dfa

        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i,sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)
        red_trans_funct = {}
        for s0 in self.dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        return red_trans_funct


    def eval_automa_acceptance(self, automa_implementation = 'logic_circuit', verbose = False, threshold= None):
        if automa_implementation in ['dfa', "dfa-cut", 'dfa.cut-sym']:
            temp = 0.00001
        else:
            temp = 1
        train_accuracy = eval_accuracy(self.classifier, self.deepAutoma, self.train_img_seq, self.train_acceptance_img, automa_implementation=automa_implementation, temp=temp, verbose=verbose, threshold=threshold)
        test_accuracy_clss= 0#eval_accuracy( self.classifier, self.deepAutoma, self.test_img_seq_clss, self.test_acceptance_img_clss, automa_implementation=automa_implementation, temp=temp, verbose=verbose, threshold=threshold)
        test_accuracy_aut= 0#eval_accuracy( self.classifier, self.deepAutoma, self.test_img_seq_aut, self.test_acceptance_img_aut, automa_implementation=automa_implementation, temp=temp, verbose=verbose, threshold=threshold)
        test_accuracy_hard= eval_accuracy( self.classifier, self.deepAutoma, self.test_img_seq_hard, self.test_acceptance_img_hard, automa_implementation=automa_implementation, temp=temp, verbose=verbose, threshold=threshold)

        return train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard

    def eval_image_classification(self):
        train_acc = eval_image_classification_from_traces(self.train_img_seq, self.train_traces, self.classifier, self.mutually_exclusive)
        test_acc = eval_image_classification_from_traces(self.test_img_seq_hard, self.test_traces, self.classifier, self.mutually_exclusive)
        return train_acc, test_acc


    def train_all(self, num_of_epochs):
        if self.automa_implementation == 'logic_circuit':
            recurrent_nn = "NS"
            old_num_states = self.deepAutoma.numb_of_states
        elif self.automa_implementation == "lstm":
            recurrent_nn = "DL"
        elif self.automa_implementation == "gru":
            recurrent_nn = "GRU"
        elif self.automa_implementation == 'transformer':
            recurrent_nn = "TRANSF"

        train_file = open(self.log_dir+self.formula_name+"_train_acc_"+recurrent_nn+"_exp"+str(self.exp_num), 'w')
        test_clss_file = open(self.log_dir+self.formula_name+"_test_clss_acc_"+recurrent_nn+"_exp"+str(self.exp_num), 'w')
        test_aut_file = open(self.log_dir+self.formula_name+"_test_aut_acc_"+recurrent_nn+"_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.formula_name+"_test_hard_acc_"+recurrent_nn+"_exp"+str(self.exp_num), 'w')
        #image_classification_train_file = open(self.log_dir+self.formula_name+"_image_classification_train_acc_"+recurrent_nn+"_exp"+str(self.exp_num), 'w')
        #image_classification_test_file = open(self.log_dir+self.formula_name+"_image_classification_test_acc_"+recurrent_nn+"_exp"+str(self.exp_num), 'w')
        self.classifier.to(device)
        self.deepAutoma.to(device)

        crossentropy = torch.nn.CrossEntropyLoss().to(device)
        print("_____________training the classifier+DFA_____________")
        old_loss = 0

        if self.automa_implementation == "logic_circuit":
            params = list(self.classifier.parameters()) + [self.deepAutoma.trans_prob] + [self.deepAutoma.rew_matrix]
            #params = list(self.classifier.fc1.parameters())+ list(self.classifier.fc2.parameters()) + [self.deepAutoma.trans_prob] + [self.deepAutoma.rew_matrix]
        else:
            params = list(self.classifier.parameters()) + list(self.deepAutoma.parameters())
        optimizer = torch.optim.Adam(params=params, lr=0.0005)
        #optimizer = torch.optim.Adam(params=params, lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer)

        batch_size = 64
        tot_size = len(self.train_img_seq)

        temp = 1
        decay = 1
        min_temp = 0.00001

        classification_temperature = 0.5

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            loss_values = []
            temp = max(temp*decay, min_temp)
            for i in range(len(self.train_img_seq)):
              image_dataset = self.train_img_seq[i].to(device)
              dataset_acceptance = self.train_acceptance_img[i].to(device)
              tot_size = len(dataset_acceptance)
              for b in range(math.ceil(tot_size / batch_size)):
                start = batch_size * b
                end = min(batch_size * (b + 1), tot_size)
                batch_image_dataset = image_dataset[start:end]
                batch_acceptance = dataset_acceptance[start:end]

                batch, length, channels, pixels1, pixels2 = batch_image_dataset.size()

                optimizer.zero_grad()

                if self.automa_implementation == "logic_circuit":
                    sym_sequence = self.classifier(batch_image_dataset.view(-1,channels, pixels1, pixels2), classification_temperature)
                    _, prediction = self.deepAutoma(sym_sequence.view(batch, length, -1), temp=temp)
                    loss = crossentropy(prediction[:, -1, :].to(device), batch_acceptance)

                else:
                    sym_sequence = self.classifier(batch_image_dataset.view(-1,channels, pixels1, pixels2))
                    prediction = self.deepAutoma(sym_sequence.view(batch, length, -1))
                    loss = crossentropy(prediction, batch_acceptance)

                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())

            new_loss =  mean(loss_values)
            print("loss: ", new_loss)
            #scheduler.step(mean(loss_values))

            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(automa_implementation=self.automa_implementation)
            if epoch % 10 == 0:
                print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                     test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))

            if self.automa_implementation == "logic_circuit":
                #train_image_classification_accuracy, test_image_classification_accuracy = self.eval_image_classification()
                #print("IMAGE CLASSIFICATION: train accuracy : {}\ttest accuracy : {}".format(train_image_classification_accuracy,test_image_classification_accuracy))

                train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(
                        automa_implementation='dfa')
                if epoch % 10 == 0:
                    print(
                        "SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(
                            train_accuracy,
                            test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))


            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))


            if mean(loss_values) <= 0.01:
                break
            if abs(old_loss - new_loss) < 0.00001 and epoch > 200:
                break
            old_loss = new_loss

        for e in range(epoch, num_of_epochs):
            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))

        #save models
        model_name = self.models_dir + self.formula_name + "_exp" + str(self.exp_num)

        torch.save(self.classifier , model_name+"_"+recurrent_nn+"_classifier.pth")
        torch.save(self.deepAutoma , model_name+"_"+recurrent_nn+"_temporal.pth")

    def minimizeDFA(self, CNN_path = None, dfa_path = None, mode = "full" ):
        if CNN_path != None:
            self.classifier = torch.load(CNN_path)
        if dfa_path != None:
            self.deepAutoma = torch.load(dfa_path)
        old_threshold = -1
        print("_______________________DFA MINIMIZATION MODE: ", mode)
        #Trasitions pruning
        if mode == "full":
            min_thre = 0
            max_thre = 1
            threshold = 0
            old_three = 1
            original_train_accuracy, _, _, _ =  self.eval_automa_acceptance(automa_implementation='dfa')
            print("original train accuracy: ", original_train_accuracy)

            while True:
                print("########### threshold: ", threshold)
                deepauto = deepcopy(self.deepAutoma)
                old_threshold = deepauto.cut_unlikely_transitions(threshold=threshold)
                print(threshold, old_threshold)
                new_train_acc = eval_accuracy(self.classifier, deepauto, self.train_img_seq, self.train_acceptance_img, automa_implementation="dfa", temp=0.00001)

                print("New train acc: ", new_train_acc)
                if new_train_acc > original_train_accuracy:
                    break
                if new_train_acc < original_train_accuracy:
                    #step to the left
                    max_thre = old_threshold
                else:
                    #step to the right
                    min_thre = old_threshold

                threshold = min_thre + (max_thre - min_thre)/2
                print(f"\n{abs(old_threshold - threshold)}\n")
                if abs(old_threshold - threshold) <= 0.001 and new_train_acc == original_train_accuracy:
                    break

            self.deepAutoma = deepcopy(deepauto)

        #dfa extraction+state minimization
        self.dfa = self.deepAutoma.net2dfa(0.00001)

        #symbols-states minimimization
        if mode == "full":
            self.dfa, deleted_syms, transformed_alphabet = minimize_dfa_symbols_and_states(self.dfa)

        #save
        self.dfa.to_graphviz().render(self.automata_dir + self.formula_name + "_exp" + str(self.exp_num) + "_minimized_"+mode+".dot")

        #print statistics
        train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(
            automa_implementation='dfa')
        print(
            "SEQUENCE CLASSIFICATION: train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(
                 train_accuracy,
                test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))
        print("num of symbols: ", len(self.dfa.alphabet))
        print("num of states: ", len(self.dfa._states))

    def train_classifier_crossentropy(self, num_of_epochs):
        train_file = open(self.log_dir+self.formula_name+"_train_acc_DL_exp"+str(self.exp_num), 'w')
        test_clss_file = open(self.log_dir+self.formula_name+"_test_clss_acc_DL_exp"+str(self.exp_num), 'w')
        test_aut_file = open(self.log_dir+self.formula_name+"_test_aut_acc_DL_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.formula_name+"_test_hard_acc_DL_exp"+str(self.exp_num), 'w')
        print("_____________training classifier+lstm_____________")
        loss_crit = torch.nn.CrossEntropyLoss()
        params = [self.classifier.parameters(), self.deepAutoma.parameters()]
        params = itertools.chain(*params)
        optimizer = torch.optim.Adam(params=params, lr=0.001)
        batch_size = 64
        tot_size = len(self.train_img_seq)
        self.classifier.to(device)
        self.deepAutoma.to(device)

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            img_i =0
            for b in range(math.floor(tot_size/batch_size)):
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_image_dataset = self.train_img_seq[start:end]
                batch_acceptance = self.train_acceptance_img[start:end]
                optimizer.zero_grad()
                losses = torch.zeros(0 ).to(device)


                for i in range(len(batch_image_dataset)):
                    img_sequence =batch_image_dataset[i].to(device)
                    target = batch_acceptance[i]
                    target = torch.LongTensor([target]).to(device)
                    sym_sequence = self.classifier(img_sequence)
                    acceptance = self.deepAutoma.predict(sym_sequence)
                    # Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    loss = loss_crit(acceptance.unsqueeze(0), target)
                    losses = torch.cat((losses, loss.unsqueeze(dim=0)), 0)

                loss = losses.mean()
                loss.backward()
                optimizer.step()
                #print("batch {}\tloss {}".format(b, loss))
            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(automa_implementation='lstm')
            print("__________________________train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                 test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))


            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))




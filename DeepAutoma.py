import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import dot2pythomata, transacc2pythomata
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import string
import random


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)

class ProbabilisticAutoma(nn.Module):
    #numb_of_actions = numb_of_symbols
    #numb_of_rewards = numb_of_automa_outputs

    def __init__(self, numb_of_actions, numb_of_states, numb_of_rewards, initial_state = 0):
        super(ProbabilisticAutoma, self).__init__()
        self.initial_state = initial_state
        self.numb_of_actions = numb_of_actions
        self.alphabet = [str(i) for i in range(numb_of_actions)]
        self.numb_of_states = numb_of_states
        self.numb_of_rewards = numb_of_rewards
        self.reward_values = torch.Tensor(list(range(numb_of_rewards)))
        self.activation = sftmx_with_temp
        #TODO: rimettere stdev a 0.1
        self.trans_prob = torch.normal(0, 1, size=( numb_of_actions, numb_of_states, numb_of_states), requires_grad=True, device=device)

        self.rew_matrix = torch.normal(0, 1, size=( numb_of_states, numb_of_rewards), requires_grad=True, device=device)


    #input: sequence of actions (batch, length_seq, num_of_actions)
    def forward(self, action_seq, temp = 1, current_state= None, verbose = False):
        if verbose:
            print("temp: ", temp)
        batch_size = action_seq.size()[0]
        length_size = action_seq.size()[1]

        pred_states = torch.zeros((batch_size, length_size, self.numb_of_states))
        pred_rew = torch.zeros((batch_size, length_size, self.numb_of_rewards))

        if current_state == None:
            s = torch.zeros((batch_size,self.numb_of_states)).to(device)
            #initial state is 0 for construction
            s[:,self.initial_state] = 1.0
        else:
            s = current_state
        if verbose:
            print("INITIAL state: ", s[0])
        for i in range(length_size):
            a = action_seq[:,i, :]


            s, r = self.step(s, a, temp, verbose)
            if verbose:
                print("Reward: ", r[0])
                print("current state: ", s[0])
            #s = sftmx(s)
            pred_states[:,i,:] = s
            pred_rew[:,i,:] = r

        return pred_states, pred_rew

    def step(self, state, action, temp, verbose = False):
        
        #if type(action) == int:
        #    action= torch.IntTensor([action])
        #activation
        if verbose:
            print("current symbol: ", action[0])
        trans_prob = self.activation(self.trans_prob, temp)
        if verbose:
            print("temp:", temp)
            print("transition matrix :", trans_prob)
        #rew_matrix = self.activation(self.rew_matrix, temp)
        #no activation
        #trans_prob = self.trans_prob
        rew_matrix = self.rew_matrix
      
        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)

        selected_prob = torch.matmul(state, trans_prob)
        if verbose:
            print("state: ", state[0])
            print("state x trans prob: ", selected_prob[0])

        next_state = torch.matmul(action.unsqueeze(1), selected_prob.squeeze())
      
        next_reward = torch.matmul(next_state, rew_matrix)
       
        return next_state.squeeze(1), next_reward.squeeze(1)


    def cut_unlikely_transitions(self, symbol_wise = False, threshold = None):
        prob_values = torch.max(self.activation(self.trans_prob, 1), dim=2)[0]
        if threshold == 0:
            print("TRANSPROB_ mean:{}\tmin:{}\tmax:{}\tstdev:{}".format(torch.mean(prob_values).item(),
                                                                        torch.min(prob_values).item(),
                                                                        torch.max(prob_values).item(),
                                                                        torch.std(prob_values).item()))

            mean = torch.mean(prob_values)
            mean_max_threshold = mean + ((torch.max(prob_values) - mean) / 2)

            #calculate KDE
            # Conversione del tensore in array NumPy
            dati = prob_values.cpu().detach().view(-1).numpy()

            # Calcola l'istogramma per analizzare la distribuzione
            #counts, bin_edges = np.histogram(dati, bins=20, density=True)
            #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centri dei bin

            # Calcolo della KDE
            kde = gaussian_kde(dati)
            bin_centers = np.linspace(dati.min(), dati.max(), 1000)
            counts = kde(bin_centers)

            # Trova i picchi nell'istogramma/kde
            peaks, _ = find_peaks(counts)


            # Trova il minimo tra i picchi
            if len(peaks) >= 2:
                # Indici dei due picchi principali
                peak1 = min(peaks)
                peak2 = max(peaks)
                # Cerca il minimo tra i due picchi
                valley_index = np.argmin(counts[peak1:peak2]) + peak1
                threshold = bin_centers[valley_index]
                print("peaks")
            else:
                threshold = mean_max_threshold.cpu().detach()#.item()
                print("mean_max")


            # PLOTTING

            plt.clf()

            #sns.kdeplot(dati, fill=True, color='blue')
            plt.plot(bin_centers, counts, color='blue', label='Curva KDE')
            plt.axvline(x=threshold, color='red')
            plt.hist(dati, bins=20, color='blue', alpha=0.7, label='Istogramma', density=True)
            if len(peaks) >= 2:
                    plt.axvline(x=bin_centers[peak1], color='green', linestyle='--',
                                label=f'Picco 1 = {bin_centers[peak1]:.2f}')
                    plt.axvline(x=bin_centers[peak2], color='purple', linestyle='--',
                                label=f'Picco 2 = {bin_centers[peak2]:.2f}')

            plt.title("Probability density estimation")
            plt.xlabel("Value")
            plt.ylabel("Density")
            #plt.legend()
            plt.grid(True)
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            plt.savefig(f"max_prob_values_distribution_{random_string}.png")
            plt.clf()

        print("Threshold: ", threshold)
        #CUT TRANSITIONS
        if symbol_wise:
            with torch.no_grad():
                # Calcola il valore massimo per ogni coppia (azione, stato)
                #max_probs = torch.max(self.trans_prob, dim=2).values  # Dimensioni: (numb_of_actions, numb_of_states)
                max_probs = prob_values  # Dimensioni: (numb_of_actions, numb_of_states)
                # Trova le posizioni dove max_prob < threshold
                below_threshold_mask = max_probs < threshold  # Dimensioni: (numb_of_actions, numb_of_states)
                count_states_below_threshold = below_threshold_mask.sum(dim=1)
                print("COUNTS per symbol: ", count_states_below_threshold)
                self.trans_prob[count_states_below_threshold > self.numb_of_states * 2.0/3.0] = torch.eye(self.numb_of_states, device=device)
        else:
            how_many_del = 0
            with torch.no_grad():
                for a in range(self.numb_of_actions):
                    for q in range(self.numb_of_states):
                        #max_prob = torch.max(self.trans_prob[a,q])
                        max_prob = prob_values[a,q]
                        if max_prob < threshold:
                            how_many_del += 1
                            self.trans_prob[a,q] = torch.zeros(self.numb_of_states)
                            self.trans_prob[a,q,q] = 1
                print("how many cut transitions: ", how_many_del)
        return threshold
    def net2dfa(self, min_temp):
        #activation
        #trans_prob = self.activation(self.trans_prob, min_temp)
        #rew_matrix = self.activation(self.rew_matrix, min_temp)

        #no activation
        trans_prob = self.activation(self.trans_prob, 1)
        rew_matrix = self.activation(self.rew_matrix, 1)

        max_val_trans_prob = torch.max(trans_prob, dim=2)[0]
        #print("TRANSPROB_ mean:{}\tmin:{}\tmax:{}\tstdev:{}".format( torch.mean(max_val_trans_prob).item(), torch.min(max_val_trans_prob).item(), torch.max(max_val_trans_prob).item(), torch.std(max_val_trans_prob).item()))
        #print("mean max rewmatrix:", torch.mean(torch.max(rew_matrix, dim=1)[0]).item())

        trans_prob = torch.argmax(trans_prob, dim= 2)
        rew_matrix = torch.argmax(rew_matrix, dim=1)

        #2transacc
        trans = {}
        for s in range(self.numb_of_states):
            trans[s] = {}
        acc = []
        for i, rew in enumerate(rew_matrix):
                if rew == 9:    #nel caso del dataset quantizzato a 10, il 9 è la label finale 
                    acc.append(True)
                else:
                    acc.append(False)
        for a in range(trans_prob.size()[0]):
            for s, s_prime in enumerate(trans_prob[a]):
                    trans[s][str(a)] = s_prime.item()

        #eliimnate useless symbols
        print(self.alphabet)
        for sym in set(self.alphabet):
                # if a symbol never changes the state is useless
                change_state = False
                for q in range(self.numb_of_states):
                    if trans[q][sym] != q:
                        change_state = True
                        break
                if not change_state:
                    for q in range(self.numb_of_states):
                        del trans[q][sym]
                    self.alphabet.remove(str(sym))
        print(self.alphabet)
        print(trans)

        if self.alphabet == []:
            self.alphabet.append('0')
            trans = {0:{'0':0}}
            acc = acc[0:1]

        pyautomaton = transacc2pythomata(trans, acc, self.alphabet)
        pyautomaton = pyautomaton.reachable()
        pyautomaton = pyautomaton.minimize()

        return pyautomaton


    def initFromDfa(self, reduced_dfa, outputs, weigth=10):
        with torch.no_grad():
            #zeroing transition probabilities
            for a in range(self.numb_of_actions):
                for s1 in range(self.numb_of_states):
                    for s2 in range(self.numb_of_states):
                        self.trans_prob[a, s1, s2] = 0.0

            #zeroing reward matrix
            for s in range(self.numb_of_states):
                for r in range(self.numb_of_rewards):
                    self.rew_matrix[s,r] = 0.0


        #set the transition probabilities as the one in the dfa
        for s in reduced_dfa:
            for a in reduced_dfa[s]:
                with torch.no_grad():
                    self.trans_prob[a, s, reduced_dfa[s][a]] = weigth

        #set reward matrix
        for s in range(len(reduced_dfa.keys())):
                with torch.no_grad():
                    self.rew_matrix[s, outputs[s]] = weigth

class LSTMClassifier(nn.Module):

    def __init__(self, hidden_dim, numb_of_symbols, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(numb_of_symbols, hidden_dim, num_layers=2, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.classifier = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        h0 = torch.zeros((self.lstm.num_layers, sentence.size()[0], self.lstm.hidden_size)).to(device)
        c0 = torch.zeros((self.lstm.num_layers, sentence.size()[0], self.lstm.hidden_size)).to(device)
        lstm_out, _= self.lstm(sentence, (h0, c0))

        logits = self.classifier(lstm_out[:, -1, :])

        return logits

class GRUClassifier(nn.Module):

    def __init__(self, hidden_dim, numb_of_symbols, output_size):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(numb_of_symbols, hidden_dim, num_layers=2, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.classifier = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        h0 = torch.zeros((self.gru.num_layers, sentence.size()[0], self.gru.hidden_size)).to(device)
        lstm_out, _= self.gru(sentence, h0)

        logits = self.classifier(lstm_out[:, -1, :])

        return logits

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, input_dim, num_classes, nhead=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        #print(f"Num classes: {num_classes}, d_model: {d_model}, nhead: {nhead}, num_layers: {num_layers}")
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Aggiunge codifiche posizionali
        seq_length = x.size(1)
        positional_encoding = self.generate_positional_encoding(seq_length, self.hidden_dim).to(device)
        x = self.embedding(x) + positional_encoding
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Pooling sull'asse della sequenza
        x = self.fc(x)
        return x.squeeze()

    def generate_positional_encoding(self, seq_length, d_model):
        # Genera codifiche posizionali sinusoidali
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # Aggiunge dimensione batch
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd
from utils import count_states_from_dot, count_symbols_from_dot
from statistics import mean, stdev, median
from scipy.stats import median_abs_deviation as mad

def plot_results(formula_name, plot_title, res_dir='Results/', num_exp=2, plot_legend=False, plot_dir="Plots/", aut_dir="AUtomata/", state_count=None, symbol_count = None):
    experiments_to_keep = 7
    fontsize = 20
    train_rr = []
    test_hard_rr = []
    x = []
    xDL = []
    xGRU = []
    xTRANSF = []

    train_rr_DL = []
    test_hard_rr_DL = []
    train_rr_GRU = []
    test_hard_rr_GRU = []
    train_rr_TRANSF = []
    test_hard_rr_TRANSF = []

    for i in range(num_exp):

        #risultati DL (lstm)
        with open(res_dir+formula_name + "_train_acc_DL_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [float(r) for r in train_res]
        train_rr_DL.append(train_res)

        with open(res_dir+formula_name + "_test_hard_acc_DL_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [float(r) for r in test_hard_res]
        test_hard_rr_DL.append(test_hard_res)

        #risultati GRU

        with open(res_dir+formula_name + "_train_acc_GRU_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [float(r) for r in train_res]
        train_rr_GRU.append(train_res)

        with open(res_dir+formula_name + "_test_hard_acc_GRU_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [float(r) for r in test_hard_res]
        test_hard_rr_GRU.append(test_hard_res)

        #risultati TRANSF
        with open(res_dir+formula_name + "_train_acc_TRANSF_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [float(r) for r in train_res]
        train_rr_TRANSF.append(train_res)

        with open(res_dir+formula_name + "_test_hard_acc_TRANSF_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [float(r) for r in test_hard_res]
        test_hard_rr_TRANSF.append(test_hard_res)

        #results NS
        with open(res_dir+formula_name + "_train_acc_NS_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [float(r) for r in train_res]
        train_rr.append(train_res)

        with open(res_dir+formula_name + "_test_hard_acc_NS_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [float(r) for r in test_hard_res]

        test_hard_rr.append(test_hard_res)

        #Risultati image classification NS
        '''
        with open(res_dir+formula_name + "_image_classification_train_acc_NS_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [50.0 + abs( float(r) - 50.0) for r in train_res]
        train_img_rr.append(train_res)

        with open(res_dir+formula_name + "_image_classification_test_acc_NS_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [50.0 + abs( float(r) - 50.0) for r in test_hard_res]

        test_img_rr.append(test_hard_res)
        '''
    ############# eliminate outlayers NS

    dict_exp_to_keep = {}
    for i, res in enumerate(train_rr):
        dict_exp_to_keep[i] =res[-1]

    ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

    keys = list(ordered.keys())

    keys = keys[-experiments_to_keep:]

    train_rr_no_ol = []
    test_hard_rr_no_ol = []
    state_distances = []
    sym_counts = []

    state_distances_full = []
    sym_counts_full = []

    if state_count == None:
        gt_automa_path = aut_dir+formula_name
        gt_state_count = count_states_from_dot(gt_automa_path)
    else:
        gt_state_count = state_count

    if symbol_count == None:
        gt_automa_path = aut_dir+formula_name
        gt_symbol_count = count_symbols_from_dot(gt_automa_path)
    else:
        gt_symbol_count = symbol_count

    for k in keys:
        train_rr_no_ol = train_rr_no_ol + train_rr[k]
        test_hard_rr_no_ol = test_hard_rr_no_ol + test_hard_rr[k]

        #count states and symbols no minimizarion
        path = aut_dir + formula_name + "_exp" + str(k) + "_minimized_only_states.dot"
        state_count = count_states_from_dot(path)
        symbol_count = count_symbols_from_dot(path)
        state_distances.append(abs(state_count - gt_state_count))
        sym_counts.append(abs(symbol_count - gt_symbol_count))

        #count states and symbols with minimization
        path = aut_dir + formula_name + "_exp" + str(k) + "_minimized_full.dot"
        state_count = count_states_from_dot(path)
        symbol_count = count_symbols_from_dot(path)
        state_distances_full.append(abs(state_count - gt_state_count))
        sym_counts_full.append(abs(symbol_count - gt_symbol_count))

        x = x + list(range(len(train_rr[k])))

    #write statistics num of symbols and states
    with open(plot_dir+"minimization_stats", "a") as f:
        f.write(f"_______{formula_name} only_state_mode_______\n")
        f.write(f"state distance: {mean(state_distances)}+-{stdev(state_distances)}\n")
        f.write(f"num symbols: {mean(sym_counts)}+-{stdev(sym_counts)}\n\n")
        #f.write(f"state distance: {median(state_distances)}+-{mad(state_distances)}\n")
        f.write(f"num symbols: {median(sym_counts)}+-{mad(sym_counts)}\n\n")

        f.write(f"_______{formula_name} full_mode_______\n")
        f.write(f"state_distance: {mean(state_distances_full)}+-{stdev(state_distances_full)}\n")
        f.write(f"num symbols: {mean(sym_counts_full)}+-{stdev(sym_counts_full)}\n\n")
        #f.write(f"state_distance: {median(state_distances_full)}+-{mad(state_distances_full)}\n")
        f.write(f"num symbols: {median(sym_counts_full)}+-{mad(sym_counts_full)}\n\n")

    ############# eliminate outlayers DL
    dict_exp_to_keep = {}
    for i, res in enumerate(train_rr_DL):
        dict_exp_to_keep[i] = res[-1]


    ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

    keys = list(ordered.keys())

    keys = keys[-experiments_to_keep:]

    train_rr_DL_no_ol = []
    test_hard_rr_DL_no_ol = []


    for k in keys:
        train_rr_DL_no_ol = train_rr_DL_no_ol + train_rr_DL[k]
        test_hard_rr_DL_no_ol = test_hard_rr_DL_no_ol + test_hard_rr_DL[k]
        xDL = xDL + list(range(len(train_rr_DL[k])))

    ############# eliminate outlayers TRANSF
    dict_exp_to_keep = {}
    for i, res in enumerate(train_rr_TRANSF):
        dict_exp_to_keep[i] = res[-1]


    ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

    keys = list(ordered.keys())

    keys = keys[-experiments_to_keep:]

    train_rr_TRANSF_no_ol = []
    test_hard_rr_TRANSF_no_ol = []


    for k in keys:
        train_rr_TRANSF_no_ol = train_rr_TRANSF_no_ol + train_rr_TRANSF[k]
        test_hard_rr_TRANSF_no_ol = test_hard_rr_TRANSF_no_ol + test_hard_rr_TRANSF[k]
        xTRANSF = xTRANSF + list(range(len(train_rr_TRANSF[k])))

    ############# eliminate outlayers GRU
    dict_exp_to_keep = {}
    for i, res in enumerate(train_rr_GRU):
        dict_exp_to_keep[i] = res[-1]


    ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

    keys = list(ordered.keys())

    keys = keys[-experiments_to_keep:]

    train_rr_GRU_no_ol = []
    test_hard_rr_GRU_no_ol = []


    for k in keys:
        train_rr_GRU_no_ol = train_rr_GRU_no_ol + train_rr_GRU[k]
        test_hard_rr_GRU_no_ol = test_hard_rr_GRU_no_ol + test_hard_rr_GRU[k]
        xGRU = xGRU + list(range(len(train_rr_GRU[k])))
        

    #################à plot train sequence classification
    plt.rcParams["figure.figsize"] = [7.5, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    #train
    plot_line(x.copy(), train_rr_no_ol, "DeepDFA", plot_legend)
    plot_line(xDL.copy(), train_rr_DL_no_ol, "LSTM", plot_legend)
    plot_line(xGRU.copy(), train_rr_GRU_no_ol, "GRU", plot_legend)
    plot_line(xTRANSF.copy(), train_rr_TRANSF_no_ol, "Transformer", plot_legend)
    plt.ylim(40, 100)
    plt.xlim(0, 500)

    if plot_legend:
        plt.legend( prop={"size":fontsize-6}, bbox_to_anchor =(1, 1))
    plt.title(plot_title+" (train)",  fontdict={'fontsize': fontsize+3})
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+formula_name+"_train_accuracy.png")
    plt.clf()

    ################### plot test accuracy
    #test
    fig, ax = plt.subplots()
    plot_line(x.copy(), test_hard_rr_no_ol, "DeepDFA", plot_legend)
    plot_line(xDL.copy(), test_hard_rr_DL_no_ol, "LSTM", plot_legend)
    plot_line(xGRU.copy(), test_hard_rr_GRU_no_ol, "GRU", plot_legend)
    plot_line(xTRANSF.copy(), test_hard_rr_TRANSF_no_ol, "Transformer", plot_legend)
    plt.ylim(40, 100)
    plt.xlim(0, 500)

    if plot_legend:
        plt.legend( prop={"size":fontsize-6}, bbox_to_anchor =(1, 1))
    plt.title(plot_title+" (test)",  fontdict={'fontsize': fontsize+3})
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+formula_name+"_test_accuracy.png")
    plt.clf()

    return


def plot_results_all_formulas(formulas, plot_title, dir='Results/', num_exp=2, plot_legend=False, plot_dir="Plots/", aut_dir="Automata/", state_count=None, symbol_count = None):
    experiments_to_keep = 7
    fontsize = 20
    ds_type = "hard"
    formula_name = "Mean over 20 Declare formulas"
    train_rr_no_ol_cum = []
    test_hard_rr_no_ol_cum = []
    train_rr_DL_no_ol_cum = []
    test_hard_rr_DL_no_ol_cum = []
    train_rr_GRU_no_ol_cum = []
    test_hard_rr_GRU_no_ol_cum = []
    train_rr_TRANSF_no_ol_cum = []
    test_hard_rr_TRANSF_no_ol_cum = []
    x_cum = []
    x_cum_DL = []
    x_cum_GRU = []
    x_cum_TRANSF = []
    state_distances = []
    sym_counts = []
    state_distances_full = []
    sym_counts_full = []

    for formula in formulas:
        if state_count == None:
            gt_automa_path = aut_dir + formula
            gt_state_count = count_states_from_dot(gt_automa_path)
        else:
            gt_state_count = state_count

        if symbol_count == None:
            gt_automa_path = aut_dir + formula
            gt_symbol_count = count_symbols_from_dot(gt_automa_path)
        else:
            gt_symbol_count = symbol_count

        train_rr = []

        test_hard_rr = []
        x = []

        train_rr_DL = []
        test_hard_rr_DL = []

        train_rr_GRU = []
        test_hard_rr_GRU = []


        train_rr_TRANSF = []
        test_hard_rr_TRANSF = []

        xDL = []
        xGRU = []
        xTRANSF = []

        for i in range(num_exp):

            #risultati DL (LSTM)
            with open(dir+formula + "_train_acc_DL_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [float(r) for r in train_res]
            train_rr_DL.append(train_res)

            with open(dir+formula + "_test_"+ds_type+"_acc_DL_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [float(r) for r in test_hard_res]
            test_hard_rr_DL.append(test_hard_res)

            #risultati GRU
            with open(dir+formula + "_train_acc_GRU_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [float(r) for r in train_res]
            train_rr_GRU.append(train_res)

            with open(dir+formula + "_test_"+ds_type+"_acc_GRU_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [float(r) for r in test_hard_res]
            test_hard_rr_GRU.append(test_hard_res)

            #risultati Transformers
            with open(dir+formula + "_train_acc_TRANSF_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [float(r) for r in train_res]
            train_rr_TRANSF.append(train_res)

            with open(dir+formula + "_test_"+ds_type+"_acc_TRANSF_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [float(r) for r in test_hard_res]
            test_hard_rr_TRANSF.append(test_hard_res)

            ###############################################################################
            #risultati NS
            with open(dir+formula + "_train_acc_NS_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [float(r) for r in train_res]
            train_rr.append(train_res)

            with open(dir+formula + "_test_"+ds_type+"_acc_NS_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [float(r) for r in test_hard_res]

            test_hard_rr.append(test_hard_res)

        ############# eliminate outlayers NS
        dict_exp_to_keep = {}
        for i, res in enumerate(train_rr):
            dict_exp_to_keep[i] =res[-1]

        ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

        keys = list(ordered.keys())
        keys = keys[-experiments_to_keep:]

        train_rr_no_ol = []
        test_hard_rr_no_ol = []
        train_img_rr_no_ol = []
        test_img_rr_no_ol = []

        for k in keys:
            train_rr_no_ol = train_rr_no_ol + train_rr[k]
            test_hard_rr_no_ol = test_hard_rr_no_ol + test_hard_rr[k]
            # count states and symbols no minimizarion
            path = aut_dir + formula + "_exp" + str(k) + "_minimized_only_states.dot"
            state_count = count_states_from_dot(path)
            symbol_count = count_symbols_from_dot(path)
            state_distances.append(abs(state_count - gt_state_count))
            sym_counts.append(abs(symbol_count - gt_symbol_count))

            # count states and symbols with minimization
            path = aut_dir + formula + "_exp" + str(k) + "_minimized_full.dot"
            state_count = count_states_from_dot(path)
            symbol_count = count_symbols_from_dot(path)
            state_distances_full.append(abs(state_count - gt_state_count))
            sym_counts_full.append(abs(symbol_count - gt_symbol_count))

            x = x + list(range(len(train_rr[k])))
        x_cum = x_cum + x

        ############# eliminate outlayers DL
        dict_exp_to_keep = {}
        for i, res in enumerate(train_rr_DL):
            dict_exp_to_keep[i] = res[-1]

        ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

        keys = list(ordered.keys())
        keys = keys[-experiments_to_keep:]

        train_rr_DL_no_ol = []
        test_hard_rr_DL_no_ol = []


        for k in keys:
            train_rr_DL_no_ol = train_rr_DL_no_ol + train_rr_DL[k]
            test_hard_rr_DL_no_ol = test_hard_rr_DL_no_ol + test_hard_rr_DL[k]
            xDL = xDL + list(range(len(train_rr_DL[k])))

        train_rr_no_ol_cum = train_rr_no_ol_cum + train_rr_no_ol
        test_hard_rr_no_ol_cum = test_hard_rr_no_ol_cum + test_hard_rr_no_ol
        #train_img_rr_no_ol_cum = train_img_rr_no_ol_cum + train_img_rr_no_ol
        #test_img_rr_no_ol_cum = test_img_rr_no_ol_cum + test_img_rr_no_ol
        train_rr_DL_no_ol_cum = train_rr_DL_no_ol_cum + train_rr_DL_no_ol
        test_hard_rr_DL_no_ol_cum = test_hard_rr_DL_no_ol_cum + test_hard_rr_DL_no_ol
        x_cum_DL = x_cum_DL + xDL

        ############# eliminate outlayers GRU
        dict_exp_to_keep = {}
        for i, res in enumerate(train_rr_GRU):
            dict_exp_to_keep[i] = res[-1]

        ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

        keys = list(ordered.keys())
        keys = keys[-experiments_to_keep:]

        train_rr_GRU_no_ol = []
        test_hard_rr_GRU_no_ol = []


        for k in keys:
            train_rr_GRU_no_ol = train_rr_GRU_no_ol + train_rr_GRU[k]
            test_hard_rr_GRU_no_ol = test_hard_rr_GRU_no_ol + test_hard_rr_GRU[k]
            xGRU = xGRU + list(range(len(train_rr_GRU[k])))

        #train_img_rr_no_ol_cum = train_img_rr_no_ol_cum + train_img_rr_no_ol
        #test_img_rr_no_ol_cum = test_img_rr_no_ol_cum + test_img_rr_no_ol
        train_rr_GRU_no_ol_cum = train_rr_GRU_no_ol_cum + train_rr_GRU_no_ol
        test_hard_rr_GRU_no_ol_cum = test_hard_rr_GRU_no_ol_cum + test_hard_rr_GRU_no_ol
        x_cum_GRU = x_cum_GRU + xGRU

        ############# eliminate outlayers TRANSFORMERS
        dict_exp_to_keep = {}
        for i, res in enumerate(train_rr_TRANSF):
            dict_exp_to_keep[i] = res[-1]

        ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

        keys = list(ordered.keys())
        keys = keys[-experiments_to_keep:]

        train_rr_TRANSF_no_ol = []
        test_hard_rr_TRANSF_no_ol = []


        for k in keys:
            train_rr_TRANSF_no_ol = train_rr_TRANSF_no_ol + train_rr_TRANSF[k]
            test_hard_rr_TRANSF_no_ol = test_hard_rr_TRANSF_no_ol + test_hard_rr_TRANSF[k]
            xTRANSF = xTRANSF + list(range(len(train_rr_TRANSF[k])))

        #train_img_rr_no_ol_cum = train_img_rr_no_ol_cum + train_img_rr_no_ol
        #test_img_rr_no_ol_cum = test_img_rr_no_ol_cum + test_img_rr_no_ol
        train_rr_TRANSF_no_ol_cum = train_rr_TRANSF_no_ol_cum + train_rr_TRANSF_no_ol
        test_hard_rr_TRANSF_no_ol_cum = test_hard_rr_TRANSF_no_ol_cum + test_hard_rr_TRANSF_no_ol
        x_cum_TRANSF = x_cum_TRANSF + xTRANSF

    #################à plot TRAIN sequence classification
    plt.rcParams["figure.figsize"] = [7.5, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    #train
    plot_line(x_cum.copy(), train_rr_no_ol_cum, "DeepDFA", plot_legend)
    plot_line(x_cum_DL.copy(), train_rr_DL_no_ol_cum, "LSTM", plot_legend)
    plot_line(x_cum_GRU.copy(), train_rr_GRU_no_ol_cum, "GRU", plot_legend)
    plot_line(x_cum_TRANSF.copy(), train_rr_TRANSF_no_ol_cum, "Transformer", plot_legend)
    plt.ylim(40, 100)
    plt.xlim(0, 500)

    if plot_legend:
        plt.legend( prop={"size":fontsize-6}, bbox_to_anchor =(1, 1))
    plt.title(plot_title+" (train)",  fontdict={'fontsize': fontsize+3})
    plt.xlabel("Epochs", fontsize=fontsize-2)
    plt.ylabel("Accuracy", fontsize=fontsize-2)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+plot_title+" train.png")
    plt.clf()

    ################### plot TEST sequence classification
    plt.rcParams["figure.figsize"] = [7.5, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    #test
    plot_line(x_cum.copy(), test_hard_rr_no_ol_cum, "DeepDFA", plot_legend)
    plot_line(x_cum_DL.copy(), test_hard_rr_DL_no_ol_cum, "LSTM", plot_legend)
    plot_line(x_cum_GRU.copy(), test_hard_rr_GRU_no_ol_cum, "GRU", plot_legend)
    plot_line(x_cum_TRANSF.copy(), test_hard_rr_TRANSF_no_ol_cum, "Transformer", plot_legend)
    plt.ylim(40, 100)
    plt.xlim(0, 500)

    if plot_legend:
        plt.legend( prop={"size":fontsize-6}, bbox_to_anchor =(1, 1))
    plt.title(plot_title+" (test)",  fontdict={'fontsize': fontsize+3})
    plt.xlabel("Epochs", fontsize=fontsize-2)
    plt.ylabel("Accuracy", fontsize=fontsize-2)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+plot_title+" test.png")
    plt.clf()

    #write statistics num of symbols and states
    with open(plot_dir+"minimization_stats", "a") as f:
        f.write(f"_______{formula_name} only_state_mode_______\n")
        f.write(f"state distance: {mean(state_distances)}+-{stdev(state_distances)}\n")
        f.write(f"num symbols: {mean(sym_counts)}+-{stdev(sym_counts)}\n\n")
        f.write(f"state distance: {median(state_distances)}+-{mad(state_distances)}\n")
        f.write(f"num symbols: {median(sym_counts)}+-{mad(sym_counts)}\n\n")

        f.write(f"_______{formula_name} full_mode_______\n")
        f.write(f"state_distance: {mean(state_distances_full)}+-{stdev(state_distances_full)}\n")
        f.write(f"num symbols: {mean(sym_counts_full)}+-{stdev(sym_counts_full)}\n\n")
        f.write(f"state_distance: {median(state_distances_full)}+-{mad(state_distances_full)}\n")
        f.write(f"num symbols: {median(sym_counts_full)}+-{mad(sym_counts_full)}\n\n")

    return


def plot_line(x, y, label, plot_legend=False):
    data = [x, y]
    print(np.array(x).shape)
    print(np.array(y).shape)
    data = np.array(data)
    data = data.T
    data = pd.DataFrame(data, columns=['x', 'y'])
    if plot_legend:
        line = seaborn.lineplot( x = data['x'],y=data['y'], label=label)
    else:
        line = seaborn.lineplot( x = data['x'],y=data['y'])
    return

def state_distance_one_formula(formula_name, aut_dir='Automata/', num_exp=2, out_file="Plots/states_distance", state_count= None, trans_cut=False):
    exp_to_keep = 7
    if state_count == None:
        gt_automa_path = aut_dir+formula_name
        gt_count = count_states_from_dot(gt_automa_path)
    else:
        gt_count = state_count
    distances = []

    for i in range(num_exp):
        if trans_cut:
            path = aut_dir+formula_name + "_exp"+str(i)+"_minimized_cut.dot"
        else:
            path = aut_dir+formula_name + "_exp"+str(i)+"_minimized.dot"
        count = count_states_from_dot(path)
        distances.append(abs(count - gt_count))

    distances = sorted(distances)
    distances = distances[:exp_to_keep]
    with open(out_file, "a") as dist_file:
        dist_file.write(formula_name)
        if trans_cut:
            dist_file.write(" TRANS CUT")
        dist_file.write("\n{}+-{}\n\n".format(mean(distances), stdev(distances) ))

def state_distance_all_formulas(formula_names, aut_dir='Automata/', num_exp=2, out_file="Plots/states_distance"):
    exp_to_keep = 7
    all_distances = []

    for formula_name in formula_names:
        gt_automa_path = aut_dir+formula_name
        gt_count = count_states_from_dot(gt_automa_path)
        distances = []

        for i in range(num_exp):
            path = aut_dir+formula_name + "_exp"+str(i)+"_minimized.dot"
            count = count_states_from_dot(path)
            distances.append(abs(count - gt_count))

        distances = sorted(distances)
        distances = distances[:exp_to_keep]
        all_distances += distances

    with open(out_file, "a") as dist_file:
            dist_file.write("ALL FORMULAS")
            dist_file.write("\n{}+-{}\n\n".format(mean(all_distances), stdev(all_distances) ))


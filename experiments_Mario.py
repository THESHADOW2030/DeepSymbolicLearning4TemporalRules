from DSLMooreMachine import DSLMooreMachine
import random
import absl.flags
import absl.app

from utils import set_seed

from flloat.parser.ltlf import LTLfParser
from create_dataset_mario import  loadMarioDataset_balanced_labels
import torchvision

from declare_formulas import formulas, formulas_names, max_length_traces
from plot import plot_results, plot_results_all_formulas, state_distance_one_formula, state_distance_all_formulas
import os

from tqdm import tqdm
from contextlib import redirect_stdout
import datetime

#flags
absl.flags.DEFINE_integer("NUM_OF_SYMBOLS", 10, "number of symbols used to initialize the model")
absl.flags.DEFINE_integer("NUM_OF_STATES", 100, "number of states used to initialize the model") #TODO: rimettere a 25 
absl.flags.DEFINE_integer("NUM_LABELS", 10, "number of labels in the dataset, used to quantize the labels")

absl.flags.DEFINE_string("LOG_DIR", "Results_mario/", "path to save the results")
absl.flags.DEFINE_string("PLOTS_DIR", "Plots_mario/", "path to save the plots")
absl.flags.DEFINE_string("AUTOMATA_DIR", "Automata_mario/", "path to save the learned automata")
absl.flags.DEFINE_string("MODELS_DIR", "Models_mario/", "path to save the learned automata")
absl.flags.DEFINE_string("DATASET_DIR", "observation_clean_labels_10", "path to the dataset directory")


FLAGS = absl.flags.FLAGS

def test_method(automa_implementation, formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, num_exp, epochs=500, log_dir="Results/", models_dir="Modles/", automata_dir="Automata/"):
    if automa_implementation == "logic_circuit":
        model = DSLMooreMachine(formula, formula_name, dfa, symbolic_dataset,image_seq_dataset, FLAGS.NUM_OF_SYMBOLS, FLAGS.NUM_OF_STATES, dataset="mario", automa_implementation= automa_implementation, num_exp=num_exp, log_dir=log_dir, models_dir=models_dir, automata_dir=automata_dir, num_labels=FLAGS.NUM_LABELS)
    else:
        if automa_implementation == 'transformer':
            model = DSLMooreMachine(formula, formula_name, dfa, symbolic_dataset,image_seq_dataset, FLAGS.NUM_OF_SYMBOLS, FLAGS.NUM_OF_STATES + 2, dataset="mario",  automa_implementation= automa_implementation, num_exp=num_exp, log_dir=log_dir, models_dir=models_dir, num_labels=FLAGS.NUM_LABELS)
        else:
            model = DSLMooreMachine(formula, formula_name, dfa, symbolic_dataset,image_seq_dataset, FLAGS.NUM_OF_SYMBOLS, FLAGS.NUM_OF_STATES, dataset="mario",  automa_implementation= automa_implementation, num_exp=num_exp, log_dir=log_dir, models_dir=models_dir, num_labels=FLAGS.NUM_LABELS)

    model.train_all(epochs)

    if automa_implementation == "logic_circuit":
        model.minimizeDFA(mode="only_states")
        model.minimizeDFA(mode="full")

############################################# EXPERIMENTS ######################################################################
def main(argv):
    if not os.path.isdir(FLAGS.LOG_DIR):
        os.makedirs(FLAGS.LOG_DIR)
    if not os.path.isdir(FLAGS.PLOTS_DIR):
        os.makedirs(FLAGS.PLOTS_DIR)
    if not os.path.isdir(FLAGS.AUTOMATA_DIR):
        os.makedirs(FLAGS.AUTOMATA_DIR)
    if not os.path.isdir(FLAGS.MODELS_DIR):
        os.makedirs(FLAGS.MODELS_DIR)

    ############ load dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    X_train, y_train =loadMarioDataset_balanced_labels(f"./{FLAGS.DATASET_DIR}/Train", transform=transform, num_outputs=2)
    X_test, y_test =loadMarioDataset_balanced_labels(f"./{FLAGS.DATASET_DIR}/Test", transform=transform, num_outputs=2)
    print("Dataset loaded with {} training samples and {} test samples.".format(len(X_train), len(X_test)))
    num_exp = 10
    if FLAGS.NUM_OF_SYMBOLS == 5:
        formula_name = "Mario Bias"
        plot_legend = False
    else:
        formula_name = "Mario No Bias"
        plot_legend = True
    formula = "Mario Game"
    print("{}_______________________".format(formula_name))

    ##### IMAGE traces dataset
    # training dataset
    print("Training dataset")
    train_img_seq, train_acceptance_img = X_train, y_train
    print("Test dataset")
    # test clss
    test_img_seq_clss, test_acceptance_img_clss = X_test, y_test
    # test_aut
    test_img_seq_aut, test_acceptance_img_aut = X_test, y_test
    # test_hard
    test_img_seq_hard, test_acceptance_img_hard = X_test, y_test

    dfa = None
    symbolic_dataset = None

    image_seq_dataset = (train_img_seq, train_acceptance_img, test_img_seq_clss, test_acceptance_img_clss, test_img_seq_aut, test_acceptance_img_aut, test_img_seq_hard, test_acceptance_img_hard)

    if (FLAGS.NUM_OF_STATES + 2) % 4 != 0:
        print("Number of states must be a multiple of 4 for the transformer implementation. Adjusting to the nearest multiple of 4.")
        FLAGS.NUM_OF_STATES += (4 - (FLAGS.NUM_OF_STATES + 2) % 4)
    if FLAGS.NUM_OF_STATES < 4:
        print("Number of states must be at least 4. Setting to 4.")
        FLAGS.NUM_OF_STATES = 4

    print(f"Number of symbols: {FLAGS.NUM_OF_SYMBOLS}, Number of states: {FLAGS.NUM_OF_STATES}")


    now = datetime.datetime.now()
    with tqdm(range(num_exp)) as pbar:
        for i in pbar:
            set_seed(9+i)
            
            
            print(f"Experiment {i}/{num_exp - 1} started at {now.strftime('%Y-%m-%d--%H:%M:%S')}")

            log_dir = FLAGS.LOG_DIR + f"NUM_LABEL_{FLAGS.NUM_LABELS}/" + now.strftime('%Y-%m-%d--%H:%M:%S') + f"_Symbols{FLAGS.NUM_OF_SYMBOLS}_States{FLAGS.NUM_OF_STATES}" "/"
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            pbar.set_description(f"Experiment {i}/{num_exp - 1} started at {now.strftime('%Y-%m-%d %H:%M:%S')}")

            print("###################### NEW TEST ###########################")
            print("formula = {},\texperiment = {}".format(formula_name, i))
            #with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
            #DeepDFA
            test_method("logic_circuit", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=log_dir, automata_dir=FLAGS.AUTOMATA_DIR, models_dir=FLAGS.MODELS_DIR)
            #lstm
            test_method("lstm", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=log_dir, models_dir=FLAGS.MODELS_DIR)
            #gru
            test_method("gru", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=log_dir, models_dir=FLAGS.MODELS_DIR)
            #transformers
            test_method("transformer", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=log_dir, models_dir=FLAGS.MODELS_DIR)
    
    
    print(f"All experiments completed at {now.strftime('%Y-%m-%d--%H:%M:%S')}")
    print("Plotting results...")

    plot_dir = FLAGS.PLOTS_DIR + f"NUM_LABEL_{FLAGS.NUM_LABELS}/" + now.strftime('%Y-%m-%d--%H:%M:%S') + f"_Symbols{FLAGS.NUM_OF_SYMBOLS}_States{FLAGS.NUM_OF_STATES}" "/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    
    automata_dir = FLAGS.AUTOMATA_DIR + f"NUM_LABEL_{FLAGS.NUM_LABELS}/" + now.strftime('%Y-%m-%d--%H:%M:%S') + f"_Symbols{FLAGS.NUM_OF_SYMBOLS}_States{FLAGS.NUM_OF_STATES}" "/"
    if not os.path.isdir(automata_dir):
        os.makedirs(automata_dir, exist_ok=True)
    models_dir = FLAGS.MODELS_DIR + f"NUM_LABEL_{FLAGS.NUM_LABELS}/" + now.strftime('%Y-%m-%d--%H:%M:%S') + f"_Symbols{FLAGS.NUM_OF_SYMBOLS}_States{FLAGS.NUM_OF_STATES}" "/"
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    plot_results_all_formulas(formulas, formulas_names, max_length_traces, log_dir=log_dir, num_exp=num_exp, plot_legend=plot_legend, plot_dir=plot_dir, aut_dir=automata_dir, state_count=FLAGS.NUM_OF_STATES, symbol_count=FLAGS.NUM_OF_SYMBOLS)


if __name__ == '__main__':
    absl.app.run(main)



from DSLMooreMachine import DSLMooreMachine
import random
import absl.flags
import absl.app
from utils import set_seed

from flloat.parser.ltlf import LTLfParser
from create_dataset import create_traces_set_one_true_literal_balanced, create_traces_set_one_true_literal_balanced_batches, create_image_sequence_dataset
import torchvision

from declare_formulas import formulas, formulas_names
from plot import plot_results, plot_results_all_formulas, state_distance_one_formula, state_distance_all_formulas
import os

#flags
absl.flags.DEFINE_integer("NUM_OF_SYMBOLS", 10, "number of symbols used to initialize the model")
absl.flags.DEFINE_integer("NUM_OF_STATES", 60, "number of states used to initialize the model") #TODO: rimettere a 25

absl.flags.DEFINE_integer("MAX_LENGTH_TRAIN_TRACES", 15, "maximum traces length used to create the train dataset")
absl.flags.DEFINE_integer("LENGTH_TEST_TRACES", 15, "maximum traces length used to create the test dataset")
absl.flags.DEFINE_integer("HALF_OF_TRAIN_TRACES", 30, "number of positive (/negative) traces used for training")
absl.flags.DEFINE_string("LOG_DIR", "Results/", "path to save the results")
absl.flags.DEFINE_string("PLOTS_DIR", "Plots/", "path to save the plots")
absl.flags.DEFINE_string("AUTOMATA_DIR", "Automata/", "path to save the learned automata")
absl.flags.DEFINE_string("MODELS_DIR", "Models/", "path to save the learned automata")
absl.flags.DEFINE_boolean("TRAIN_ON_RESTRICTED_DATASET", True, "if True, train on a restricted dataset (MNIST)")


FLAGS = absl.flags.FLAGS

def test_method(automa_implementation, formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, num_exp, epochs=500, log_dir="Results/", models_dir="Modles/", automata_dir="Automata/"):
    if automa_implementation == "logic_circuit":
        model = DSLMooreMachine(formula, formula_name, dfa, symbolic_dataset,image_seq_dataset, FLAGS.NUM_OF_SYMBOLS, FLAGS.NUM_OF_STATES, automa_implementation= automa_implementation, num_exp=num_exp, log_dir=log_dir, models_dir=models_dir, automata_dir=automata_dir)
    else:
        model = DSLMooreMachine(formula, formula_name, dfa, symbolic_dataset,image_seq_dataset, FLAGS.NUM_OF_SYMBOLS, FLAGS.NUM_OF_STATES, automa_implementation= automa_implementation, num_exp=num_exp, log_dir=log_dir, models_dir=models_dir)

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
    #take the images
    normalize = torchvision.transforms.Normalize(mean=(0.1307,),
                                                     std=(0.3081,))

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    if FLAGS.TRAIN_ON_RESTRICTED_DATASET:
            test_data = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
            train_data = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)
    else:
            train_data = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
            test_data = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)

    num_exp = 10

    if FLAGS.NUM_OF_SYMBOLS == 2:
        plot_title = "MNIST Bias"
    else:
        plot_title = f"MNIST No Bias ({FLAGS.NUM_OF_SYMBOLS})"

    for i in range(len(formulas)):
        formula = formulas[i]
        formula_name = formulas_names[i]

        print("{}_______________________".format(formula_name))
        # translate the formula (used to create the dataset)
        parser = LTLfParser()
        ltl_formula_parsed = parser(formula)
        dfa = ltl_formula_parsed.to_automaton()
        alphabet = ["c" + str(i) for i in range(2)]
        ###############create the dataset
        #set the seed for the random split of the dataset
        random.seed(1)

        ##### SYMBOLIC traces dataset
        train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = create_traces_set_one_true_literal_balanced(FLAGS.MAX_LENGTH_TRAIN_TRACES, FLAGS.LENGTH_TEST_TRACES, FLAGS.HALF_OF_TRAIN_TRACES, alphabet, dfa, verbose=True)
        symbolic_dataset = (train_traces, test_traces, train_acceptance_tr, test_acceptance_tr)


        ##### IMAGE traces dataset
        # training dataset
        print("Training dataset")
        train_img_seq, train_acceptance_img = create_image_sequence_dataset(train_data, 2, train_traces,
                                                                                          train_acceptance_tr, print_size=True)

        print("Test dataset")
        # test clss
        test_img_seq_clss, test_acceptance_img_clss = create_image_sequence_dataset(test_data, 2,
                                                                                                  train_traces,
                                                                                                  train_acceptance_tr, print_size=True)
        # test_aut
        test_img_seq_aut, test_acceptance_img_aut = create_image_sequence_dataset(train_data, 2,
                                                                                                test_traces,
                                                                                                test_acceptance_tr, print_size=True)
        # test_hard
        test_img_seq_hard, test_acceptance_img_hard = create_image_sequence_dataset(test_data, 2,
                                                                                                  test_traces,
                                                                                                  test_acceptance_tr, print_size=True)

        image_seq_dataset = (train_img_seq, train_acceptance_img, test_img_seq_clss, test_acceptance_img_clss, test_img_seq_aut, test_acceptance_img_aut, test_img_seq_hard, test_acceptance_img_hard)

        for i in range(num_exp):
            #set_seed
            set_seed(9+i)
            print("###################### NEW TEST ###########################")
            print("formula = {},\texperiment = {}".format(formula_name, i))

            #DeepDFA
            test_method("logic_circuit", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=FLAGS.LOG_DIR, automata_dir=FLAGS.AUTOMATA_DIR, models_dir=FLAGS.MODELS_DIR)
            #lstm
            test_method("lstm", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=FLAGS.LOG_DIR, models_dir=FLAGS.MODELS_DIR)
            #gru
            test_method("gru", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=FLAGS.LOG_DIR, models_dir=FLAGS.MODELS_DIR)
            #transformers
            test_method("transformer", formula, formula_name, dfa, symbolic_dataset, image_seq_dataset, i, log_dir=FLAGS.LOG_DIR, models_dir=FLAGS.MODELS_DIR)

        plot_results(formula_name, formula_name, res_dir = FLAGS.LOG_DIR,num_exp=num_exp, plot_legend=True, plot_dir= FLAGS.PLOTS_DIR,  aut_dir=FLAGS.AUTOMATA_DIR)
        state_distance_one_formula(formula_name, FLAGS.AUTOMATA_DIR, num_exp, FLAGS.PLOTS_DIR+"state_distance")

    plot_results_all_formulas(formulas_names, plot_title, dir=FLAGS.LOG_DIR, num_exp=num_exp,plot_legend=True, plot_dir= FLAGS.PLOTS_DIR, aut_dir=FLAGS.AUTOMATA_DIR)

if __name__ == '__main__':
    absl.app.run(main)



# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from robustbench import load_model
from robustbench.data import load_cifar10
from autoattack import AutoAttack
from robustbench.eval import benchmark

parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument(
    "base_classifier", type=str, help="path to saved pytorch model of base classifier"
)
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument(
    "--split", choices=["train", "test"], default="test", help="train or test set"
)
parser.add_argument("--N0", type=int, default=5)//10
parser.add_argument("--N", type=int, default=10, help="number of samples to use")//20
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()
#1)train, test, cerify, autoattack
if __name__ == "__main__":

    print("Prova CUDA")
    print(torch.cuda.is_available())  # Deve stampare True se CUDA è installato correttamente
    print(torch.cuda.get_device_name(0))  # Stampa il nome della GPU
    print("verifico prova")
    # load the base classifier
    checkpoint = torch.load(args.base_classifier) #controlla
    base_classifier = load_model(model_name='Sehwag2021Proxy_R18', dataset='cifar10', threat_model='L2')
    # Copiamo lo state_dict esistente
    state_dict = checkpoint["state_dict"]

    # Creiamo un nuovo dizionario con le chiavi aggiornate
    new_state_dict = {}

    # Iteriamo su ogni chiave nello state_dict originale
    for key in state_dict.keys():
        # Rimuoviamo il prefisso '1.' dalle chiavi, se presente
        new_key = key.replace('1.', '', 1)  # replace solo la prima occorrenza
        # Aggiorniamo il nuovo dizionario con la chiave modificata e il valore originale
        new_state_dict[new_key] = state_dict[key]

    # Aggiorniamo il checkpoint con il nuovo state_dict
    checkpoint["state_dict"] = new_state_dict

    # Stampo le nuove chiavi per verificarle
    for key in checkpoint["state_dict"].keys():
        print(key)
    base_classifier.load_state_dict(checkpoint["state_dict"])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(
        base_classifier, get_num_classes(args.dataset), args.sigma
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clean_acc, robust_acc = benchmark(smoothed_classifier.base_classifier,
                                      dataset='cifar10',
                                      threat_model='L')

    """

    x_test, y_test = load_cifar10(n_examples=50)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    adversary = AutoAttack(base_classifier, norm='L2', eps= 0.5)
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)


    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        #da qui inizia la certificazione----

        adversary = AutoAttack(model, norm='Linf', eps=8 / 255, version='custom',
                               attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_test, y_test)

        
        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        prediction, radius = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch
        )
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed
            ),
            file=f,
            flush=True,
        )
        

    f.close()

"""
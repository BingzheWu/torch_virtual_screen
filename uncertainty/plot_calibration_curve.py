import argparse
import os
import numpy as np
#np.set_printoptions(precision=2)
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--num_bins", type=int, default=20, help="bin number for ECE")
args = parser.parse_args()
def parse(npz_arr):
    return npz_arr["predictions"], npz_arr["targets"]

def calibration_curve(npz_arr):
    outputs, labels = parse(npz_arr)
    if outputs is None:
        out = None
    else:
        confidences = np.max(outputs, 1)
        step = (confidences.shape[0] + args.num_bins - 1) // args.num_bins
        bins = np.sort(confidences)[::step]
        bins = bins[:args.num_bins]
        print(bins)
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
            predictions = np.argmax(outputs, 1)
            bin_lowers = bins[:-1]
            bin_uppers = bins[1:]

            accuracies = predictions==labels
            xs = []
            max_xs = []
            ys = []
            zs = []

            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) * (confidences < bin_upper)
                prop_in_bin = in_bin.mean()
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    max_confidence_in_bin = confidences[in_bin].min()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    xs.append(avg_confidence_in_bin)
                    max_xs.append(max_confidence_in_bin)
                    ys.append(accuracy_in_bin)
                    zs.append(prop_in_bin)
            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)

            out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece, "max_conf": max_xs}
        return out

def get_path(filename, append="", no_npz=False):
    path = os.path.join(args.path, filename + append)
    if not no_npz:
        path += ".npz"
    print(path)
    return path

def plot():
    result_tox21_gcn = np.load('logs/HIV_GCN_over_sampling/HIV_HIV_activetest.npz')
    result_tox21_mpnn = np.load('logs/HIV_GCN_normal/HIV_HIV_activetest.npz')
    methods = [
        (result_tox21_gcn, "GCN"),
        (result_tox21_mpnn, 'GCN-normal')
    ]
    results = dict()
    x = np.arange(args.num_bins)
    fig, ax = plt.subplots()
    for method, name in methods:
        out = calibration_curve(method)
        print("Number of Positive samples {}".format(method['targets'].sum()))
        print(method['predictions'].shape)
        print(method['targets'].shape)
        ax.plot(x, (out['confidence'] - out['accuracy'])[:args.num_bins], "o-", label=name)
        print(out['ece'])
    #plt.show()
    ax.legend()
    plt.xticks(x[::3], np.round(out['confidence'][::3], decimals=3))
    plt.hlines(0, 0, args.num_bins-1, colors="k", linestyles="dashed")
    plt.xlabel('Conficdence (average)')
    plt.ylabel('Confidence-Accuracy')
    plt.grid(True)
    fig.savefig("test.pdf", dpi=150)

if __name__=='__main__':
    plot()
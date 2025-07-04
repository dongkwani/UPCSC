"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import os
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden

# METRICS = ['accuracy','auroc_in','aupr_in','aupr_out','fpr95']
# METRICS = ['accuracy', 'macro_f1']
METRICS = ['accuracy']

def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    # print("===")
    #print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:  # seeds
        fpath = None
        
        log_files = os.listdir(osp.join(directory, subdir))
        log_files = [file for file in log_files if 'log' in file]
        log_files.sort(reverse=True)
        
        for idx in range(len(log_files)):
            temp_fpath = osp.join(directory, subdir, log_files[idx])
            with open(temp_fpath, "r") as f:
                lines = f.read()
                if "Finish training" in lines:
                    fpath = temp_fpath
        
        # assert check_isfile(fpath)
        if fpath is None or not check_isfile(fpath):
            continue

        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)
    print("%-20s" %outputs[0]["file"].split("/")[-3], end="")
    for output in outputs:
        msg = ""
        for key, value in output.items():
            if key=="file":
                continue
            if isinstance(value, float):
                # msg += f"{key}: {value:.4f}. "
                msg += f"{value:.4f}"
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        # print(msg)
        print(msg, end=" ")
    print("")

    output_results = OrderedDict()
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        # print(f"* average {key}: {avg:.4f}% +- {std:.4f}%")
        output_results[key] = avg
    # print("===")

    return output_results


def main(args, end_signal):
    final_results = defaultdict(list)
    
    for metric_enum in METRICS:
        metric = {
            "name": metric_enum,
            "regex": re.compile(fr"\* {metric_enum}: ([\.\deE+-]+)%"),
        }

        print("===============")
        print(f"Metric: {metric_enum}")
        # results_list = []
        
        if args.multi_exp:
            # final_results = defaultdict(list)

            for directory in listdir_nohidden(args.directory, sort=True):
                directory = osp.join(args.directory, directory)
                results = parse_function(
                    metric, directory=directory, args=args, end_signal=end_signal
                )

                for key, value in results.items():
                    final_results[key].append(value)
                    # results_list.append(value)

            # print("Average performance")
            # for key, values in final_results.items():
            #     avg = np.mean(values)
            #     print(f"* {key}: {avg:.4f}%")

        else:
            parse_function(
                metric, directory=args.directory, args=args, end_signal=end_signal
            )
    
    if len(METRICS) > 1:
        print("")
        print("===Averages===")
        for key, values in final_results.items():
            avg = np.mean(values)
            round_values = [round(el,4) for el in values]
            print(f"* {key}:\t {round_values}, Total: {avg:.4f}")

if __name__ == "__main__":
    # li = ['accuracy','auroc_in','aupr_in','aupr_out','fpr95']
    # for i in range(5):
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95",
        action="store_true",
        help=r"compute 95\% confidence interval"
    )
    parser.add_argument(
        "--test-log", action="store_true", help="parse test-only logs"
    )
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword",
        default="accuracy",
        type=str,
        help="which keyword to extract"
    )
    args = parser.parse_args()

    end_signal = "=> result"  # needs to be adapted to the latest
    if args.test_log:
        end_signal = "=> result"

    args.multi_exp = True
    main(args, end_signal)


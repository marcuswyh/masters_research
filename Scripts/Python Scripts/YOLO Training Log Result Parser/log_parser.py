# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 20:28
# @Author  : Adesun
# @Site    : https://github.com/Adesun
# @File    : log_parser.py

import argparse
import logging
import os
import platform
import re
import sys
import itertools
import numpy as np

# set non-interactive backend default when os is not windows
if sys.platform != 'win32':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def show_message(message, stop=False):
    print(message)
    if stop:
        sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="training log parser by DeepKeeper ")
    parser.add_argument('--source-dir', dest='source_dir', type=str, default=r'C:\Users\Marcus Wong\Downloads\CIT Msc AI\MASTERS RESEARCH\DarkNet\darknet',
                        help='the log source directory')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default=r'C:\Users\Marcus Wong\Downloads\CIT Msc AI\MASTERS RESEARCH\DarkNet\darknet\results',
                        help='the directory to be saved')
    parser.add_argument('--csv-file', dest='csv_file', type=str, default="",
                        help='training log file')
    parser.add_argument('--log-file', dest='log_file', type=str, default="log.txt",
                        help='training log file')
    parser.add_argument('--show', dest='show_plot', type=bool, default=True,
                        help='whether to show')
    return parser.parse_args()


def log_parser(args):
    if not args.log_file:
        show_message('log file must be specified.', True)

    log_path = os.path.join(args.source_dir, args.log_file)
    if not os.path.exists(log_path):
        show_message('log file does not exist.', True)

    file_name, _ = get_file_name_and_ext(log_path)
    log_content = open(log_path).read()

    iterations = []
    losses = []
    fig, ax = plt.subplots()

    calc_map = True
    if calc_map:
        
        # set area we focus on
        ax.set_ylim(0, 100)
        map_pattern = re.compile(r"next mAP calculation at (\b\d+\b) iterations")
        map_match = (list(dict.fromkeys(map(int, map_pattern.findall(log_content)))))[:-1]

        map_acc_pattern = re.compile(r"mAP@0.5 = (\d+\.\d+)")
        map_acc_match = list(dict.fromkeys(map(float, map_acc_pattern.findall(log_content))))

        normalized = np.array(map_acc_match) * (8/100)

        plt.plot(np.array(map_match), normalized, linestyle='-', marker='o', label="mAP (%)")
        plt.xlabel('Iteration')
        plt.ylabel('mAP Accuracy')
        plt.tight_layout()
        for i,j in zip(map_match,map_acc_match):
            display = (8/100) * j
            ax.annotate(str(j)+"%",xy=(i,display))

        patt1 = re.compile(r"for conf_thresh = \d+\.\d+, precision = (\d+\.\d+), recall = (\d+\.\d+), F1-score = (\d+\.\d+)")
        patt2 = re.compile(r"for conf_thresh = \d+\.\d+, TP = (\d+), FP = (\d+), FN = (\d+), average IoU = (\d+\.\d+)")
        match = patt1.findall(log_content)
        match2 = patt2.findall(log_content)

        csv_path = os.path.join(args.save_dir, 'results.csv')
        out_file = open(csv_path, 'w')
        iters = map_match
        mAP = map_acc_match

        for i in range(len(match)):
            precision = (match[i])[0]
            recall = (match[i])[1]
            f1 = (match[i])[2]

            TP = (match2[i])[0]
            FP = (match2[i])[1]
            FN = (match2[i])[2]
            IoU = (match2[i])[3]

            iteration = iters[i]
            AP = mAP[i]

            out_file.write(str(iteration) + ',' + TP + ',' + FP + ',' + FN + ',' + precision + ',' + recall + ',' + f1 + ',' + IoU + ',' + str(AP) + '\n')

        
    
    calc_map = False

    if calc_map == False:
        ax.set_ylim(0, 8)
        major_locator = MultipleLocator()
        minor_locator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.grid(True, which='minor')


    pattern = re.compile(r"([\d].*): .*?, (.*?) avg")
    matches = pattern.findall(log_content)
    counter = 0
    log_count = len(matches)

    if args.csv_file != '':
        csv_path = os.path.join(args.save_dir, args.csv_file)
        out_file = open(csv_path, 'w')
    else:
        csv_path = os.path.join(args.save_dir, file_name + '.csv')
        out_file = open(csv_path, 'w')

    for match in matches:
        counter += 1
        if log_count > 200:
            if counter % 200 == 0:
                print('parsing {}/{}'.format(counter, log_count))
        else:
            print('parsing {}/{}'.format(counter, log_count))
        iteration, loss = match
        iterations.append(int(iteration))
        losses.append(float(loss))
        out_file.write(iteration + ',' + loss + '\n')

    ax.plot(iterations, losses, label = "Loss scores")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend()

    # saved as svg
    save_path = os.path.join(args.save_dir, file_name + '.svg')
    plt.savefig(save_path, dpi=300, format="svg")
    if args.show_plot:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    log_parser(args)

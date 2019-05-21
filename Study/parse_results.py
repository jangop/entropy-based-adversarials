#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pandas


# TODO: Quick and dirty ;)
def parse_result_json(file_in):
    with open(file_in, 'r') as f_in:
        result = json.load(f_in)

        data = result['data']
        df = pandas.DataFrame(data, columns=["id", "y","t"])
        df = df.sort_values(["id"]).reset_index(drop=True) 

        return df

def add_ground_truth_and_entropy(df, ground_truth_file_in):
    df_ground_truth = pandas.read_csv(ground_truth_file_in)
    df["truth"] = df_ground_truth["image_a_is_original"]
    df["entropy"] = df_ground_truth["entropy"]
    df["method"] = df_ground_truth["method"]

    return df


def process_files(files_in, groud_truth_file_in):
    df_final = pandas.DataFrame(columns=["user_id", "id", "y", "t"])
     
    i  = 0
    for file_in in files_in:
        df = parse_result_json(file_in)
        df['user_id'] = [i for _ in range(len(df))]
        i += 1

        df = add_ground_truth_and_entropy(df, groud_truth_file_in)

        df_final = df_final.append(df, ignore_index=True)

    return df_final


def list_files(path, ext='.json'):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isfile(os.path.join(path, d)) and d.endswith(ext)]


if __name__ == "__main__":
    files = list_files("Results", ".json")
    df = process_files(files, "Webpage/mturk/data/groundtruth.csv")
    df["truth"] = df["truth"].apply(lambda x: 0 if x == -1 else 1)  # We do not need left/right information!
    df.to_csv("Results/results.csv", header=True, sep=',', index=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import pandas


# TODO: Quick and dirty ;)
def csv_to_json(file_in, file_out):
        with open(file_out, "w") as f_out:
                f_out.write("{\"data\":[")

                df = pandas.read_csv(file_in)
                columns = df.columns

                first = True
                for _, row in df.iterrows():
                        data = "{" if first is True else ",{"

                        first = False
                        i = 0
                        for c in columns:
                                data += "\"{0}\":\"{1}\"".format(c, row[c])
                                i += 1

                                if i < len(columns):
                                        data += ","
                        data += "}"

                        f_out.write(data)

                f_out.write("]}")


if __name__ == "__main__":
    csv_to_json("batch.csv", "batch.json")
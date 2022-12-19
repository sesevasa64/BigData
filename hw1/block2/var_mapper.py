#!/usr/bin/python3
import csv
import sys


def main():
    vals = []
    # Предварительно был убран хедер из csv-шки
    reader = csv.reader(sys.stdin)
    for fields in reader:
        price = int(fields[9])
        vals.append(price)
    chunk_size = len(vals)
    chunk_mean = sum(vals) / chunk_size
    chunk_var = sum([(v - chunk_mean) ** 2 for v in vals]) / chunk_size
    print("{} {} {}".format(chunk_size, chunk_mean, chunk_var))    


if __name__ == "__main__":
    main()

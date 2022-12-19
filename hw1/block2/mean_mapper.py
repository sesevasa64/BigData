#!/usr/bin/python3
import csv
import sys


def main():
    acc = 0
    # Предварительно был убран хедер из csv-шки
    reader = csv.reader(sys.stdin)
    for i, fields in enumerate(reader, start=1):
        price = int(fields[9])
        acc += price
    chunk_size = i
    chunk_mean = acc / chunk_size
    print("{} {}".format(chunk_size, chunk_mean))    


if __name__ == "__main__":
    main()

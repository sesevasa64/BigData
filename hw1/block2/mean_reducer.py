#!/usr/bin/python3
import sys


def main():
    acc_size = 0
    acc_mean = 0
    for line in sys.stdin:
        chunk_size, chunk_mean = line.split()
        chunk_size = int(chunk_size)
        chunk_mean = float(chunk_mean)
        acc_mean = (chunk_size * chunk_mean + acc_size * acc_mean) / (chunk_size + acc_size)
        acc_size += chunk_size
    print("{} {}".format(acc_size, acc_mean))    


if __name__ == "__main__":
    main()

#!/usr/bin/python3
import sys


def main():
    acc_size = 0
    acc_mean = 0
    acc_var = 0
    for line in sys.stdin:
        chunk_size, chunk_mean, chunk_var = line.split()
        chunk_size = int(chunk_size)
        chunk_mean = float(chunk_mean)
        chunk_var = float(chunk_var)
        new_size = chunk_size + acc_size
        acc_var = (chunk_size * chunk_var + acc_size * acc_var) / new_size + \
            chunk_size * acc_size * ((chunk_mean - acc_mean) / new_size) ** 2
        acc_mean = (chunk_size * chunk_mean + acc_size * acc_mean) / new_size
        acc_size += chunk_size
    print("{} {} {}".format(acc_size, acc_mean, acc_var))    


if __name__ == "__main__":
    main()

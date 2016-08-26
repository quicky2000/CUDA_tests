#!/bin/sh
nvcc -g -G -lineinfo $1 -std=c++11 --ptxas-options=-v -DNDEBUG
#EOF

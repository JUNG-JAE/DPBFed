#!/bin/bash

round="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

for var in $round
do
	python3 main.py -net 'resnet18' -gpu
	sleep 1
done

#!/bin/bash

# python -m visdom.server &

pip install sewar

if [ ! -d "results" ]; then
	mkdir results
else
	if [ ! -d "results/Training" ]; then
		mkdir results/Training
	fi
	if [ ! -d "results/Evaluation" ]; then
		mkdir results/Evaluation
	fi
fi

if [ ! -d "weights" ]; then
	mkdir weights
fi

if [ ! -d "runs" ]; then
	mkdir runs
fi

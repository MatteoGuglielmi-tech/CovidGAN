#!/bin/bash

# function that checks if the arguments are correct
function checkArguments() {
	if [ -z "$1" ] || [ -z "$2" ]; then
		echo "Error: missing arguments."
		echo "Usage: ./setup.sh <pkg manager> (pip/conda) <check-pytorch> (y/n) "
		return 1
	fi

	return 0
}

function folders() {
	if [ ! -d "results" ]; then
		mkdir ../results
	else
		if [ ! -d "results/Training" ]; then
			mkdir ../results/Training
		fi
		if [ ! -d "results/Evaluation" ]; then
			mkdir ../results/Evaluation
		fi
	fi

	if [ ! -d "weights" ]; then
		mkdir ../weights
	fi

	if [ ! -d "runs" ]; then
		mkdir ../runs
	fi
}

function sewar() {
	pip install sewar
}

function deps() {
	if [ "$1" == "pip" ]; then
		pip install -r dependencies.yml
	else
		conda env create -f dependencies.yml
	fi
}

function check_pytorch() {
	if [ "$1" == "y" ]; then
		python ./pytorch-sanity-check.py
	fi
}

main() {
	if checkArguments "$1" "$2"; then
		deps "$1"
		check_pytorch "$2"
		sewar
		folders
	else
		echo "Quitting..."
		exit 1
	fi
}

echo "Starting pipeline..."
main "$1" "$2"

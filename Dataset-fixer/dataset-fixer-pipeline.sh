#!/bin/bash

# This script aims to filter the original dataset from .avi files
# The script takes two arguments:
#  - the path to the folder to be cleaned
#  - the path to the folder where the .avi files will be moved
#  - the path to the folder where the .mat files will be moved

# function that checks if the arguments are correct
function checkArguments() {
	if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
		echo "Error: missing arguments."
		echo "Usage: ./dataFiltering.sh <path to the folder to be cleaned> <path to the folder where the .avi files will be moved>
        <path to the folder where the .mat files will be moved>"
		return 1
	fi

	if [ ! -d "$1" ]; then
		echo "Error: $1 is not a valid path."
		return 1
	fi

	if [ ! -d "$2" ]; then
		echo "Error: $2 is not a valid path."
		return 1
	fi

	if [ ! -d "$3" ]; then
		echo "Error: $3 is not a valid path."
		return 1
	fi

	return 0
}

# function that deletes all .avi file from current folder and subfolders maintaining the folder structure
function move-avi-files() {
	for file in "$1"/*; do
		if [ -d "$file" ]; then
			move-avi-files "$file" "$2"
		else
			if [[ "$file" == *.avi ]]; then
				mv "$file" "$2"
				echo "File $file moved to $2"
			fi
		fi
	done
}

# function that deletes all the files in the MATLAB folder
# to be called before the conversion of .mat files
# it pops a multiple choice question to the user to confirm the deletion of the files
function empty-MATLAB-Folder() {
	if [ ! -d "$1" ]; then
		mkdir "$1"
	else
		if [[ "$(ls -A "$1")" ]]; then
			echo "Warning: $1 is not empty. All the files in $1 will be deleted. Do you want to continue? (y/n) (Digit the corresponding number and press enter))"
			select yn in "Yes" "No"; do
				case $yn in
				Yes)
					find "$1" -type f -delete -print
					break
					;;
				No) exit ;;
				*) echo "Please answer 1 or 2." ;;
				esac
			done
		fi
	fi
}

# function that converts .mat files to .csv and .png
function convertMat() {
	reg4scores='.*score.mat'
	reg4frames='.*[0-9]+.mat'
	reg4Imagesfolder='.*Images'
	for file in "$1"/*; do
		if ! [[ "$file" == "SanMatteo.mat" ]]; then
			if [[ -d "$file" ]]; then
				if ! [[ $file =~ $reg4Imagesfolder ]]; then
					convertMat "$file" "$2"
				else
					continue
				fi
			else
				if [[ $file =~ $reg4scores ]]; then
					python3 python-tools/mat2csv-converter.py "$file"
					mv "$file" "$2/Scores/"
				elif [[ $file =~ $reg4frames ]]; then
					python3 python-tools/mat2png-converter.py "$file"
					mv "$file" "$2/Frames/"
				else
					continue
				fi
			fi
		else
			continue
		fi
	done
}

# function that creates the folders for the scores in the dataset {0...3}
# to be called before organizerRunner
function scoreFoldersCreator() {
	current=$(pwd)
	cd "$1" || exit
	for i in {0..3}; do
		mkdir "$i"
	done
	cd "$current" || exit
}

# function that moves the images in the correct folder corresponding to the score
function organizerRunner() {
	reg4images='.*.png'
	for file in "$1"/*/*/*; do
		# check if the current OS is macOS or Linux and use the corresponding grep command
		if [[ "$OSTYPE" == "darwin"* ]]; then
			currentDir=$(ggrep -Po '(?<=-[0-9]/)[0-9]{1}$' <<<"$file")
		elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
			currentDir=$(grep -Po '(?<=-[0-9]/)[0-9]{1}' <<<"$file")
		else
			exit 1
		fi
		if [[ -d "$file" ]]; then
			if [[ "$currentDir" == "0" || "$currentDir" == "1" || "$currentDir" == "2" || "$currentDir" == "3" ]]; then
				continue
			else
				organizerRunner "$file"
			fi
		else
			if [[ $file =~ $reg4images ]]; then
				python3 python-tools/organizer.py "$file"
			else
				continue
			fi
		fi
	done
}

function resizeImages() {
	for file in "$1"/*; do
		if [[ -d "$file" ]]; then
			python3 python-tools/resizer.py "$file"
		else
			continue
		fi
	done
}

function centerImages() {
	binaryMask='.*Binary-Masks.*'
	for file in "$1"/*; do
		if [[ -d "$file" ]]; then
			echo "$file"
			if [[ $file =~ $binaryMask ]]; then
				continue
			else
				centerImages "$file"
			fi
		else
			python3 python-tools/mask.py "$file"
		fi
	done
}

function moveImages() {
	# check if Imags folder in Backup folder exists
	if [ ! -d "../../Backup/Images" ]; then
		mkdir "../../Images/Images"
	fi
	if [ ! -d "../../Dati-San-Matteo-2/Binary-Masks" ]; then
		mv ../../Dati-San-Matteo-2/Binary-Masks/ ../../Images/
	fi
	mv ../../Dati-San-Matteo-2/1017/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1045/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1047/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1048/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1050/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1051/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1052/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1066/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1067/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1068/ ../../Backup-Folder/Images/
	mv ../../Dati-San-Matteo-2/1069/ ../../Backup-Folder/Images/
}

function createImageFolderStruct() {
	mkdir "$1/0-root"
	mv "$1/0" "$1/0-root"
	mkdir "$1/1-root"
	mv "$1/1" "$1/1-root"
	mkdir "$1/2-root"
	mv "$1/2" "$1/2-root"
	mkdir "$1/3-root"
	mv "$1/3" "$1/3-root"
}

# wrapper function to perform pipeline
main() {
	if checkArguments "$1" "$2" "$3"; then
		move-avi-files "$1" "$3"
		empty-MATLAB-Folder "$2"
		convertMat "$1" "$2"
		scoreFoldersCreator "$1"
		organizerRunner "$1"
		moveImages
		resizeImages "$1"
		centerImages "$1"
		createImageFolderStruct "$1"
	else
		echo "Quitting..."
		exit 1
	fi
}

echo "Starting pipeline..."
main "$1" "$2" "$3"

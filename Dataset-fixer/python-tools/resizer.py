"""Script that normalizes images to the biggest image dimensions found in a folder."""
import sys

from utils import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resizer.py <input_folder>")
        sys.exit(1)
    else:
        _, path = read_arguments(howmany=2)
        logger = logger_setup()
        logger.info(f"Inspecting folder {path}...")
        files = extractFolderContent(folder=path)
        logger.info(
            "Folder content succesfully extracted. Proceeding to extract biggest image dimensions..."
        )
        maxH, maxW = biggestImageDimensions(files=files, path=path)
        logger.info(
            f"Biggest image dimensions: {maxH}x{maxW} found. Proceeding to resize..."
        )
        padding(files=files, path=path, finalDims=(maxH, maxW))

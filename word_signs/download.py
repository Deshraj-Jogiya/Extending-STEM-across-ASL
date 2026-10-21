"""Download the word-sign clips from the WLASL dataset (Hugging Face mirror Voxel51/WLASL).

Usage: python -m word_signs.download <data_dir>
Writes <data_dir>/selection.json (the vocabulary and every WLASL clip of those signs) and the clips under <data_dir>/data/.
WLASL is licensed for non-commercial computational use only (C-UDA); the clips are not redistributed in this repository.
"""

import json
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "Voxel51/WLASL"
VOCABULARY = ["computer", "internet", "email", "program", "network", "machine", "print", "television", "write", "letter", "paper",
              "test", "learn", "study", "teacher", "school", "science", "math", "calculate", "research", "measure", "plus",
              "energy", "chemistry", "biology", "physics", "experiment", "book", "picture"]


def main(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    index = hf_hub_download(REPO, "samples.json", repo_type="dataset", local_dir=data_dir)
    samples = json.load(open(index, encoding="utf-8"))["samples"]
    clips = [{"filepath": s["filepath"], "gloss": s["gloss"]["label"]} for s in samples if s["gloss"]["label"] in VOCABULARY]
    with open(os.path.join(data_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"vocabulary": VOCABULARY, "clips": clips}, f, indent=1)
    for clip in clips:
        hf_hub_download(REPO, clip["filepath"], repo_type="dataset", local_dir=data_dir)
    print(f"{len(clips)} clips of {len(VOCABULARY)} signs in {data_dir}")


if __name__ == "__main__":
    main(sys.argv[1])

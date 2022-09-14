import argparse
import logging
import os, errno
import pickle

import datasets

from huggingface_hub import HfApi
from transformers import AutoTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# 21 -> 31 / BS 64
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--nb_datasets", type=int, default=2, help="The number of datasets you want to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--index", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--bin_file_path",
        type=str,
        default="/home/younesbelkada/disk/data/bloom-data/train",
        help="The path to the binarized data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset (train, val, test)",
    )
    args = parser.parse_args()

    logger.info(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    eos = tokenizer.eos_token
    batch_size = args.batch_size
    index = args.index

    # Replace by looping over the available datasets
    api = HfApi()
    d = api.list_datasets(author="bigscience-data", use_auth_token=True)
    if index is None:
        corpus = [d.id for d in d if "roots_" in d.id][: args.nb_datasets]
    else:
        corpus = [d.id for d in d if "roots_" in d.id][index:index+args.nb_datasets]


    logger.info("Start encoding")
    logger.info(f"{len(corpus)} datasets to process.")

    rslt_ = []
    current_batch_size = 0
    total_files = 1

    for i, corpus_name in enumerate(corpus):
        logger.info(f"Loading dataset {corpus_name}")

        json_dataset = datasets.load_dataset(corpus_name, use_auth_token=True)

        data_to_process = json_dataset[args.split]["text"]
        logger.info(f"{len(data_to_process)} lines to process.")

        final_dir_path = os.path.join(args.bin_file_path, corpus_name)
        mkdir(final_dir_path)

        prefix = corpus[i].split("/")[-1]
        for line in data_to_process:
            text = f"{line.strip()} {eos}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt_.append(token_ids)
            current_batch_size += 1
            if current_batch_size >= batch_size:
                output_file = os.path.join(final_dir_path, "processed-{}-{}.bin".format(prefix, total_files))
                logger.info(f"Dump to {output_file}")
                with open(output_file, "wb") as handle:
                    pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)
                rslt_ = []
                total_files += 1
                current_batch_size = 0

        if len(rslt_) > 0:
            output_file = os.path.join(final_dir_path, "processed-{}-{}.bin".format(prefix, total_files))
            logger.info(f"Dump to {output_file}")
            with open(output_file, "wb") as handle:
                pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        json_dataset.cleanup_cache_files()

    logger.info("End encoding")


if __name__ == "__main__":
    main()

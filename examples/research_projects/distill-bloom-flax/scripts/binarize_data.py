import argparse
import logging
import os

import pickle

import datasets

from huggingface_hub import HfApi
from transformers import AutoTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--nb_datasets", type=int, default=2, help="The number of datasets you want to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--bin_file_path", type=str, default="/home/sanchitgandhi/cache/younes_files/binarized_data", help="The path to the binarized data.")
    args = parser.parse_args()

    logger.info(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    eos = tokenizer.eos_token
    batch_size = args.batch_size

    # Replace by looping over the available datasets
    api = HfApi()
    d = api.list_datasets(author="bigscience-catalogue-lm-data", use_auth_token=True)
    corpus = [d.id for d in d if "cleaned" in d.id][:args.nb_datasets]

    logger.info(f"Loading text from {corpus} and converting to json")
    json_datasets = [datasets.load_dataset(corpus_name, use_auth_token=True) for corpus_name in corpus]

    logger.info("Start encoding")
    logger.info(f"{len(json_datasets)} datasets to process.")

    rslt_ = []
    current_batch_size = 0
    total_files = 1

    for i, data in enumerate(json_datasets):
        data_to_process = data["train"]["text"]
        logger.info(f"{len(data_to_process)} lines to process.")
        prefix = corpus[i].split("/")[-1]
        for line in data_to_process:
            text = f"{line.strip()} {eos}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt_.append(token_ids)
            current_batch_size += 1
            if current_batch_size >= batch_size:
                output_file = os.path.join(args.bin_file_path, "processed-{}-{}.bin".format(prefix, total_files))
                logger.info(f"Dump to {output_file}")
                with open(output_file, "wb") as handle:
                    pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)
                rslt_ = []
                total_files += 1
                current_batch_size = 0
        if len(rslt_) > 0:
            output_file = os.path.join(args.bin_file_path, "processed-{}-{}.bin".format(prefix, total_files))
            logger.info(f"Dump to {output_file}")
            with open(output_file, "wb") as handle:
                pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info("End encoding")

if __name__ == "__main__":
    main()



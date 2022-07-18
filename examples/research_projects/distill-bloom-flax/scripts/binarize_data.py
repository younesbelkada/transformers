import argparse
import logging
import os

import pickle

import datasets
from transformers import AutoTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--hf_dataset_path", type=str, default="/mnt/disks/persist/younes_data/raw_hf_data", help="The path to the hf dataset.")
    parser.add_argument("--hf_dataset_name", type=str, default="cleaned_lm_indic-te_ted_talks_iwslt", help="The name of the hf dataset you want to process.")
    parser.add_argument("--bin_file_path", type=str, default="/mnt/disks/persist/younes_data/binarized_data", help="The path to the binarized data.")
    args = parser.parse_args()

    logger.info(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    eos = tokenizer.eos_token

    logger.info(f"Loading text from {args.hf_dataset_path} and converting to json")
    json_dataset = datasets.load_dataset(os.path.join("bigscience-catalogue-lm-data", args.hf_dataset_name), use_auth_token=True)
    data = json_dataset["train"]["text"]

    logger.info("Start encoding")
    logger.info(f"{len(json_dataset['train']['text'])} examples to process.")

    rslt_ = []

    for line in data:
        text = f"{line.strip()} {eos}"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        rslt_.append(token_ids)

    output_file = os.path.join(args.bin_file_path, "{}.bin".format(args.hf_dataset_name))
    logger.info(f"Dump to {output_file}")
    with open(output_file, "wb") as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()



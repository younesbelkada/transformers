from argparse import ArgumentParser, Namespace
from json import dump
from os import linesep, remove
from os.path import exists
from tempfile import NamedTemporaryFile

from requests import get
from transformers import logger
from transformers.commands import BaseTransformersCLICommand
from transformers.tokenization_utils import SentencePieceExtractor, YouTokenToMeExtractor


def extract_vocab_command_factory(args: Namespace):
    return VocabCommand(args.format, args.input[0], args.vocab_output, args.merges_output)


class VocabCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        vocab_parser = parser.add_parser(
            "extract-vocab", help="CLI tool providing various helper method regarding tokenizer vocabularies."
        )

        vocab_parser.add_argument("--format", choices=["sentencepiece", "yttm"], help="Input model format")
        vocab_parser.add_argument(
            "--vocab-output", required=True, type=str, help="Path to where the extracted model vocab will be saved."
        )
        vocab_parser.add_argument(
            "--merges-output", required=True, type=str, help="Path to where the extracted model merges will be saved."
        )
        vocab_parser.add_argument("input", nargs=1, type=str, help="Path to the model to extract data from.")
        vocab_parser.set_defaults(func=extract_vocab_command_factory)

    def __init__(self, type: str, input_path: str, vocab_output_path: str, merges_output_path: str):
        self._remote_model = None

        self.input_path = input_path
        self.vocab_output_path = vocab_output_path
        self.merges_output_path = merges_output_path
        self._extractor = (
            SentencePieceExtractor(input_path) if type == "sentencepiece" else YouTokenToMeExtractor(input_path)
        )

    def run(self):
        try:
            if self.input_path.startswith("http"):
                # Saving model
                with NamedTemporaryFile("wb", delete=False) as f:
                    logger.info("Writing content from {} to {}".format(self.input_path, f.name))
                    response = get(self.input_path, allow_redirects=True)
                    f.write(response.content)

                    self._remote_model = self.input_path
                    self.input_path = f.name

            # Open output files and let's extract model information
            with open(self.vocab_output_path, "w") as vocab_f:
                with open(self.merges_output_path, "w") as merges_f:
                    # Do the extraction
                    vocab, merges = self._extractor.extract()

                    # Save content
                    dump(vocab, vocab_f)
                    merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{linesep}", merges))
        finally:
            # If model was downloaded from internet we need to cleanup the tmp folder.
            if self._remote_model is not None and exists(self.input_path):
                remove(self.input_path)

# coding=utf-8
"""IVA_MT_WSLOT Huggingface dataset custom loading script"""


import datasets
import json


_DESCRIPTION = """\
IVA_MT is a dataset for machine translation used in NLU (Virual Assistant) that can transfer slots between languages.
"""

_URL = "https://github.com/cartesinus/iva_mt/raw/main/release/0.2/iva_mt_wslot-dataset-en2es-0.2.0.tar.gz"

_LANGUAGE_PAIRS = ["en-pl", "en-de", "en-es", "en-sv"]

class IVA_MTConfig(datasets.BuilderConfig):
    """BuilderConfig for IVA_MT"""

    def __init__(self, language_pair, **kwargs):
        super().__init__(**kwargs)
        """

        Args:
            language_pair: language pair, you want to load
            **kwargs: keyword arguments forwarded to super.
        """
        self.language_pair = language_pair


class IVA_MT(datasets.GeneratorBasedBuilder):
    """OPUS-100 is English-centric, meaning that all training pairs include English on either the source or target side."""

    VERSION = datasets.Version("0.2.0")

    BUILDER_CONFIG_CLASS = IVA_MTConfig
    BUILDER_CONFIGS = [
        IVA_MTConfig(name=pair, description=_DESCRIPTION, language_pair=pair)
        for pair in _LANGUAGE_PAIRS
    ]

    def _info(self):
        src_tag, tgt_tag = self.config.language_pair.split("-")
        return datasets.DatasetInfo(
#            features=datasets.Features({"translation": datasets.features.Translation(languages=(src_tag, tgt_tag))}),
            features=datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "locale": datasets.Value("string"),
                    "origin": datasets.Value("string"),
                    "partition": datasets.Value("string"),
                    "translation_utt": datasets.features.Translation(languages=(src_tag, tgt_tag)),
                    "translation_xml": datasets.features.Translation(languages=(src_tag, tgt_tag)),
                    "src_bio": datasets.Value("string"),
                    "tgt_bio": datasets.Value("string")
                }
            ),
            supervised_keys=(src_tag, tgt_tag),
        )

    def _split_generators(self, dl_manager):

        lang_pair = self.config.language_pair
        src_tag, tgt_tag = lang_pair.split("-")

        archive = dl_manager.download(_URL)

        data_dir = "/".join(["iva_mt_wslot-dataset", "0.2.0", lang_pair])
        output = []

        test = datasets.SplitGenerator(
            name=datasets.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": f"{data_dir}/iva_mt_wslot-{lang_pair}-test.jsonl",
                "files": dl_manager.iter_archive(archive),
                "split": "test",
            },
        )

        output.append(test)

        train = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "filepath": f"{data_dir}/iva_mt_wslot-{lang_pair}-train.jsonl",
                "files": dl_manager.iter_archive(archive),
                "split": "train",
            },
        )

        output.append(train)

        valid = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": f"{data_dir}/iva_mt_wslot-{lang_pair}-valid.jsonl",
                "files": dl_manager.iter_archive(archive),
                "split": "valid",
            },
        )

        output.append(valid)

        return output

    def _generate_examples(self, filepath, files, split):
        """Yields examples."""
        src_tag, tgt_tag = self.config.language_pair.split("-")
        key_ = 0
        lang = _LANGUAGE_PAIRS.copy()

        for path, f in files:
            l = path.split("/")[-1].split("-")[1].replace('2', '-')

            if l != self.config.language_pair:
                continue

            # Read the file
            lines = f.read().decode(encoding="utf-8").split("\n")

            for line in lines:
                if not line:
                    continue

                data = json.loads(line)

                if data["partition"] != split:
                    continue

                yield key_, {
                    "id": data["id"],
                    "locale": data["locale"],
                    "origin": data["origin"],
                    "partition": data["partition"],
                    "translation_utt": {src_tag: str(data['src_utt']), tgt_tag: str(data['tgt_utt'])},
                    "translation_xml": {src_tag: str(data['src_xml']), tgt_tag: str(data['tgt_xml'])},
                    "src_bio": str(data['src_bio']),
                    "tgt_bio": str(data['tgt_bio'])
                }

                key_ += 1

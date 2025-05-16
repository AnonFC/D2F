# Enhancing Fact-Checking with LLMs: From Ambiguity Resolution to Claim Validation

## Datasets

This repository uses data from the [RawFC](https://github.com/Nicozwy/CofCED/tree/main/Datasets/RAWFC) and [LIAR](https://huggingface.co/datasets/liar) datasets. 

## Setup

### API Keys

1. Obtain an OpenAI API key and save it to the variable `OPENAI_API_KEY` in `config.py`.

2. Obtain a [SerpApi](https://serpapi.com/) key and save it to the variable `SERPAPI_KEY` in `config.py`.

### Prerequisites

```sh
python3 -m venv myenv
source myenv/bin/activate

pip install openai serpapi google-search-results sklearn
```

## Usage

```
python D2F.py --data_set RAWFC
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgments

Some code are modified from [HiSS](https://github.com/jadeCurl/HiSS), we thank their work.

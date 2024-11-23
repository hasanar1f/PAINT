# llava-hallunication-fix

The goal of this project is to mitigate the in context hallucination of LLaVA by tweaking its attention mask during token generation

## Installation and Setup

```bash
conda env create -f modPAI/environment.yml
conda activate modpai

# conda create --name hallucination python=3.10
# conda activate hallufix
# pip install -e transformers
# pip install -e lmms-eval
# pip install sentencepiece seaborn ipykernel
```

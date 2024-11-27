# llava-hallunication-fix

The goal of this project is to mitigate the in context hallucination of LLaVA by tweaking its attention mask during token generation.
The code base is based on PAI (ECCV 2024, Paper: https://arxiv.org/abs/2407.21771 and code: https://github.com/LALBJ/PAI).

## Installation and Setup

```bash
conda env create -f modPAI/environment.yml
conda activate modpai
```

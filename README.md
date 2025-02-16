###  **PAINT** (**P**aying **A**ttention to **IN**formed **T**okens)

Paper: [PAINT: PAYING ATTENTION TO INFORMED TOKENS TO MITIGATE HALLUCINATION IN LARGE VISION-LANGUAGE MODEL](https://arxiv.org/abs/2501.12206)

**Abstract:** Large Vision Language Models (LVLMs) have demonstrated remarkable capabilities in understanding and describing visual content, achieving state-of-the-art performance across various vision-language tasks. However, these models often generate descriptions containing objects or details that are absent in the input image, a phenomenon commonly known as hallucination. Our work investigates the key reasons behind this issue by analyzing the attention patterns of tokens across transformer layers and heads. We find that hallucinations often arise from the progressive weakening of attention to visual tokens in the deeper layers of the LLM. Some previous works naively boost the attention of all visual tokens to mitigate this issue, resulting in suboptimal hallucination reduction. To address this, we identify two critical sets of visual tokens that facilitate the transfer of visual information from the vision encoder to the LLM. Local tokens encode grounded information about objects present in an image, while summary tokens capture the overall aggregated representation of the image. Importantly, these two sets of tokens require different levels of attention enhancement. To this end, we propose **PAINT** (**P**aying **A**ttention to **IN**formed **T**okens), a plug-and-play framework that intervenes in the self-attention mechanism of the LLM, selectively boosting the attention of local and summary tokens with learned margins. Extensive experiments on the MSCOCO dataset demonstrate that our approach reduces hallucination rates by up to 62.3\% compared to baseline models while maintaining strong task performance. 

### Installation and Setup

```bash
conda env create -f modPAI/environment.yml
conda activate modpai
```

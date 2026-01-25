# Model-Belief  
**Token-Level Belief Extraction for Language Models**

Model-Belief is a lightweight Python library for extracting **belief distributions** from large language models (LLMs) with respect to a predefined *alternative set*.  
It operationalizes the idea that an LLM’s decision is revealed at a **specific token position**—the *pivot token*—and that the model’s uncertainty can be quantified via token-level logits.

This library accompanies the research paper:

> **From Model Choice to Model Belief: Establishing a New Measure for LLM-Based Research**
> https://arxiv.org/abs/2512.23184

---

## Citation

If you use this library or the belief-extraction methodology in your research, please cite:

```bibtex
@article{sun2025model,
  title={From model choice to model belief: Establishing a new measure for LLM-based research},
  author={Sun, Hongshen and Zhang, Juanjuan},
  journal={arXiv preprint arXiv:2512.23184},
  year={2025}
}
```
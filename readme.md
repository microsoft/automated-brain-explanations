<h3 align="center">🧠 How does the brain process language? 🧠</h3>
<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9--3.11-blue">
  <img src="https://img.shields.io/badge/numpy-%3E=2.0-blue">
</p>
<p align="center">
  We’ve been studying how to <i>scalably</i> answer this question using
  <b>LLMs</b> and <b>large-scale brain-imaging datasets</b>.
  Together, these let us automatically generate and test scientific hypotheses about language processing in the brain,
  potentially enabling a new paradigm for scientific research.
  This repo contains code for running these analyses.
</p>

<img align="center" style="width: 100%; max-width:100%;" src="https://raw.githubusercontent.com/microsoft/automated-brain-explanations/main/docs/qa_map.svg">

-----

### References

**This repo contains code underlying 2 neuroscience studies:**

<details>
<summary>Generative causal testing to bridge data-driven models and scientific theories in language neuroscience <a href="https://arxiv.org/abs/2410.00812">(Antonello*, Singh*, et al., 2024, Nature neuroscience)</a>
</summary>
<br>
Generative causal testing (GCT) is a framework for generating concise explanations of language selectivity in the brain from predictive models and then testing those explanations in follow-up experiments using LLM-generated stimuli.
</details>
<details>
<summary>Evaluating scientific theories as predictive models in language neuroscience <a href="https://www.biorxiv.org/content/10.1101/2025.08.12.669958v1">(Singh*, Antonello*, et al. 2025, bioRxiv)</a>
</summary>
<br>
QA encoding models builds features by annotating a language stimulus with the answers to yes-no questions using an LLM.
</details>
<br>

**This repo also contains code for experiments in 3 ML studies** (for a simple scikit-learn interface to use these, see [imodelsX](https://github.com/csinva/imodelsX)):
<details>
<summary>Aug-imodels: Augmenting interpretable models with large language models during training <a href="https://www.nature.com/articles/s41467-023-43713-1">(Singh et al. 2023, Nature communications)</a>
</summary>
<br>
Aug-imodels is a framework for LLMs to build extremely efficient and interpretable prediction models, e.g. linear ngram models or decision trees. Aug-imodels use LLMs during fitting but not during inference, allowing complete transparency and dramatic speed improvements.
</details>
<details>
<summary>QA-Emb: Crafting interpretable embeddings by asking LLMs questions <a href="https://arxiv.org/abs/2405.16714">(Benara*, Singh*, et al. 2024, NeurIPS)</a>
</summary>
<br>
QA-Emb is a more general version of QA encoding models, that generally builds text embeddings by asking LLMs a series of yes/no questions.
</details>
<details>
<summary>SASC: Explaining black box text modules in natural language with language models <a href="https://arxiv.org/abs/2305.09863">(Singh*, Hsu*, et al. 2023, NeurIPS workshop)</a>
</summary>
<br>
SASC is a pipeline for generating natural language explanations of black-box text modules using LLMs and synthetic causal testing.
</details>
<br>

**Finally, here are 3 studies that share the codebase here**:
<details>
<summary>Interpretable embeddings of speech enhance and explain brain encoding performance of audio models
 <a href="https://arxiv.org/abs/2507.16080">(Shimizu et al. 2025, arXiv)</a>
</summary>
<br>
Using QA-Encoding models to analyze and improve black-box speech encoding models.
</details>
<details>
<summary>Interpretable next-token prediction via the generalized induction head<a href="https://arxiv.org/abs/2411.00066">(Kim*, Mantena*, et al. 2025, NeurIPS)</a>
</summary>
<br>
Hand-engineering an induction head to retrieve features from the context can help improve interpretable fMRI encoding models.
</details>
<details>
<summary>Vector-ICL: In-context learning with continuous vector representations <a href="https://openreview.net/pdf?id=xing7dDGh3">(Zhuang et al. 2025, ICLR)</a>
</summary>
<br>
Can convert fMRI responses to continuous vector representations that can be used with LLMs to do few-shot decoding of QA features.
</details>
<br>

### Setting up

**Dataset**

- To quickstart, just download the responses / wordsequences for 3 subjects from the [encoding scaling laws paper](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230420528915)
  - This is all the data you need if you only want to analyze 3 subjects and don't want to make flatmaps
- To run full experiments, go through the paths in `neuro/config.py` and download data to the appropriate locations from [this box folder](https://utexas.box.com/s/7ur0fsr52nephxp96hs5dxm99rk2v1u0)
  - To download the main dataset here (the [HuthLab fMRI passive listening dataset](https://openneuro.org/datasets/ds003020/versions/3.1.0)), run `python experiments/00_load_dataset.py` (will download the data using [datalad](https://github.com/datalad/datalad))
- To make flatmaps, need to set [pycortex filestore](https://gallantlab.org/pycortex/auto_examples/quickstart/show_config.html) to `{root_dir}/ds003020/derivative/pycortex-db/`
- The `data/decoding` folder contains a quickstart easy example for TR-level decoding
  - It has everything needed, but if you want to visualize the results on a flatmap, you need to download the relevant PCs from [here](https://utexas.box.com/s/7ur0fsr52nephxp96hs5dxm99rk2v1u0)

**Code**

- Install using [uv](https://docs.astral.sh/uv/). Clone the repo, `cd` into the repo, run `uv add git+https://github.com/csinva/imodelsX`, then run `uv sync`. This will locally install the `neuro` package
- Useful functions
  - Loading responses
    - `neuro.data.response_utils` function `load_response`
    - Loads responses from at `{neuro.config.root_dir}/ds003020/derivative/preprocessed_data/{subject}`, where they are stored in an h5 file for each story, e.g. `wheretheressmoke.h5`
  - Loading stimulus
    - `ridge_utils.features.stim_utils` function `load_story_wordseqs`
    - Loads textgrids from `{root_dir}/ds003020/derivative/TextGrids`, where each story has a TextGrid file, e.g. `wheretheressmoke.TextGrid`
    - Uses `{root_dir}/ds003020/derivative/respdict.json` to get the length of each story

**Demo**

- `python experiments/02_fit_encoding.py`
  - Running this script with no args runs a simple small run. This script takes many relevant arguments through argparse to run different experiments

### External links

- Big thanks to folks that released open-source brain-imaging datasets, especially the [HuthLab fMRI passive listening dataset](https://openneuro.org/datasets/ds003020/versions/3.1.0) and the [Podcast ECoG dataset](https://www.nature.com/articles/s41597-025-05462-2)
- See related [fMRI experiments](https://github.com/csinva/fmri)
- Built from [this template](https://github.com/csinva/cookiecutter-ml-research)

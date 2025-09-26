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
<summary>Generative causal testing to bridge data-driven models and scientific theories in language neuroscience <a href="https://arxiv.org/abs/2410.00812">(Antonello*, Singh*, et al., 2024, arXiv)</a>
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
Recent large language models (LLMs), such as ChatGPT, have demonstrated remarkable prediction performance for a growing array of tasks. However, their proliferation into high-stakes domains and compute-limited settings has created a burgeoning need for interpretability and efficiency. We address this need by proposing Aug-imodels, a framework for leveraging the knowledge learned by LLMs to build extremely efficient and interpretable prediction models. Aug-imodels use LLMs during fitting but not during inference, allowing complete transparency and often a speed/memory improvement of greater than 1000x for inference compared to LLMs. We explore two instantiations of Aug-imodels in natural-language processing: Aug-Linear, which augments a linear model with decoupled embeddings from an LLM and Aug-Tree, which augments a decision tree with LLM feature expansions. Across a variety of text-classification datasets, both outperform their non-augmented, interpretable counterparts. Aug-Linear can even outperform much larger models, e.g. a 6-billion parameter GPT-J model, despite having 10,000x fewer parameters and being fully transparent. We further explore Aug-imodels in a natural-language fMRI study, where they generate interesting interpretations from scientific data.
</details>
<details>
<summary>QA-Emb: Crafting interpretable Embeddings by asking LLMs questions <a href="https://arxiv.org/abs/2405.16714">(Benara*, Singh*, et al. 2024, NeurIPS)</a>
</summary>
<br>
Large language models (LLMs) have rapidly improved text embeddings for a growing array of natural-language processing tasks. However, their opaqueness and proliferation into scientific domains such as neuroscience have created a growing need for interpretability. Here, we ask whether we can obtain interpretable embeddings through LLM prompting. We introduce question-answering embeddings (QA-Emb), embeddings where each feature represents an answer to a yes/no question asked to an LLM. Training QA-Emb reduces to selecting a set of underlying questions rather than learning model weights.<br>
We use QA-Emb to flexibly generate interpretable models for predicting fMRI voxel responses to language stimuli. QA-Emb significantly outperforms an established interpretable baseline, and does so while requiring very few questions. This paves the way towards building flexible feature spaces that can concretize and evaluate our understanding of semantic brain representations. We additionally find that QA-Emb can be effectively approximated with an efficient model, and we explore broader applications in simple NLP tasks.
</details>
<details>
<summary>SASC: Explaining black box text modules in natural language with language models <a href="https://arxiv.org/abs/2305.09863">(Singh*, Hsu*, et al. 2023, NeurIPS workshop)</a>
</summary>
<br>
SASC takes in a text module and produces a natural explanation for it that describes what it types of inputs elicit the largest response from the module (see Fig below). The GCT paper tests this in detail in an fMRI setting.
<br>

SASC is similar to the nice [concurrent paper](https://github.com/openai/automated-interpretability) by OpenAI, but simplifies explanations to describe the function rather than produce token-level activations. This makes it simpler/faster, and makes it more effective at describing semantic functions from limited data (e.g. fMRI voxels) but worse at finding patterns that depend on sequences / ordering.

To use, follow the instructions at <a href="https://github.com/csinva/imodelsX">imodelsX</a>, install with `pip install imodelsx` then the below shows a quickstart example.

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
- Other studies that build off the codebase here: Explaining speech encoding models ([Shimizu et al. 2025](https://arxiv.org/abs/2507.16080)), Induction-Gram ([Kim et al. 2024](https://arxiv.org/abs/2411.00066)), Vector in-context learning ([Zhuang et al. 2025](https://arxiv.org/abs/2410.05629))
- Big thanks to folks that released open-source brain-imaging datasets, especially the [HuthLab fMRI passive listening dataset](https://openneuro.org/datasets/ds003020/versions/3.1.0) and the [Podcast ECoG dataset](https://www.nature.com/articles/s41597-025-05462-2)
- See related [fMRI experiments](https://github.com/csinva/fmri)
- Built from [this template](https://github.com/csinva/cookiecutter-ml-research)

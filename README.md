# Signer-independent preprocessing for GASLT

This repository contains [Keren Artiaga’s preprocessing scripts](NEW_preprocessing) for signer-independent GASLT experiments on RWTH-PHOENIX-Weather 2014 T (PHOENIX14T). The associated [EMNLP 2025 Findings paper](https://aclanthology.org/2025.findings-emnlp.997/) evaluated three gloss-free sign language translation models, GFSLT-VLP, GASLT, and SignCL, on PHOENIX14T and CSL-Daily. The scripts here cover the PHOENIX14T preprocessing workflow for GASLT.

The workflow pools the official PHOENIX14T annotations and repartitions the clips into nine folds. Each fold holds out one signer for development and a different signer for testing. The other seven signers are used for training. Each signer serves as the test signer once.

| Script | Output |
| --- | --- |
| [`pack_dataset.py`](NEW_preprocessing/pack_dataset.py) | GASLT-format train, dev, and test files for the selected fold |
| [`name_to_video.py`](NEW_preprocessing/name_to_video.py) | A zero-based video ID mapping for the fold |
| [`cos_sim.py`](NEW_preprocessing/cos_sim.py) | A cosine similarity matrix indexed by that mapping |

## Prepare a fold

Clone this repository and run the commands from its root.

```bash
git clone https://github.com/peonycabbage/myGASLT_github.git
cd myGASLT_github
```

Download the [PHOENIX14T annotations](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and obtain the [TSPNet visual features](https://github.com/verashira/TSPNet) with window 8 and stride 2. The packer expects a flat feature directory containing one `<video_name>.pt` per annotation, with 1,024 values per timestep. Set the paths below to your local copies. The annotations directory is typically `annotations/manual` and contains the three official `PHOENIX-2014-T.<split>.corpus.csv` files.

Use a separate Python 3.10 or 3.11 environment for preprocessing. NumPy 1.x keeps the similarity pickle readable by the legacy training environment.

```bash
conda create -n gaslt-preprocess python=3.11 pip
conda activate gaslt-preprocess
python -m pip install "numpy<2" pandas torch sentence-transformers
ANNOTATIONS=/path/to/PHOENIX-2014-T/annotations/manual
FEATURES=/path/to/span=8_stride=2
FOLD=1
python NEW_preprocessing/pack_dataset.py --annotations-dir "$ANNOTATIONS" --features-dir "$FEATURES" --fold "$FOLD"
python NEW_preprocessing/name_to_video.py --annotations-dir "$ANNOTATIONS" --fold "$FOLD"
python NEW_preprocessing/cos_sim.py --annotations-dir "$ANNOTATIONS" --fold "$FOLD"
```

The last command downloads the `all-mpnet-base-v2` sentence encoder on first use. It builds pairwise translation similarities in the same order as the fold’s video IDs. The packed splits are written to `data/pht/data/signer_folds/fold_1/` and the ID mapping and similarity matrix to `data/pht/sim/`.

## Use the outputs with GASLT

Aoxiong Yin and colleagues developed the [original GASLT model and implementation](https://github.com/YinAoXiong/GASLT). Clone that repository separately and follow its setup instructions for training, evaluation, and the required BPE assets. Copy the generated `data/pht/` outputs into the GASLT checkout, preserving their directory structure. In a copy of GASLT’s `configs/train_pht.yaml`, set these values for fold 1.

| Config key | Value |
| --- | --- |
| `data.data_path` | `./data/pht/data/signer_folds/fold_1/` |
| `data.test` | `phoenix14t.pami0.test` |
| `model.sim_name_to_video_id_json` | `./data/pht/sim/fold_1.json` |
| `model.sim_video_cos_sim` | `./data/pht/sim/fold_1_cos_sim.pkl` |
| `training.model_dir` | `./results/pht/fold_1` |

For folds 2–9, repeat preprocessing with the desired fold number and update the paths accordingly. The [`gaslt-implementation` branch](https://github.com/peonycabbage/myGASLT_github/tree/gaslt-implementation) preserves the previous checkout, including its local compatibility changes and fold configuration.

## Cite the evaluation study

If you use this preprocessing workflow, cite the study by Artiaga et al.

```bibtex
@inproceedings{artiaga-etal-2025-rethinking,
    title = {Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation},
    author = {Artiaga, Keren and Kamila, Sabyasachi and Afli, Haithem and Lynch, Conor and Hasanuzzaman, Mohammed},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
    month = nov,
    year = {2025},
    address = {Suzhou, China},
    publisher = {Association for Computational Linguistics},
    pages = {18379--18391},
    doi = {10.18653/v1/2025.findings-emnlp.997},
    url = {https://aclanthology.org/2025.findings-emnlp.997/}
}
```

## Cite the GASLT model

If you use GASLT, also cite the original paper by Yin et al.

```bibtex
@inproceedings{yin2023gloss,
  title={Gloss attention for gloss-free sign language translation},
  author={Yin, Aoxiong and Zhong, Tianyun and Tang, Li and Jin, Weike and Jin, Tao and Zhao, Zhou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2551--2562},
  year={2023}
}
```

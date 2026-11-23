# Signer-independent evaluation with GASLT

This repository adapts [GASLT](https://github.com/YinAoXiong/GASLT) for signer-independent experiments on RWTH-PHOENIX-Weather 2014 T (PHOENIX14T). **Keren Artiaga’s contribution is the [signer-independent preprocessing workflow](NEW_preprocessing).** GASLT’s model architecture and original implementation are by Aoxiong Yin and colleagues (CVPR 2023).

The workflow pools the official PHOENIX14T annotations and repartitions clips into nine folds. Each fold holds out one signer for development and a different signer for testing. Training uses the other seven signers. Each signer serves as the test signer once.

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

## Train and evaluate

The [included config](configs/train_pht.yaml) targets fold 1. Place the GASLT German SentencePiece model (`de.wiki.bpe.vs25000.model`) and word vectors (`de.wiki.bpe.vs25000.d300.w2v.txt`) under `data/pht/bpe/`. See the [upstream GASLT data instructions](https://github.com/YinAoXiong/GASLT#datasets) for the original assets.

Training uses the original Linux CUDA environment in [`env.yaml`](env.yaml).

```bash
conda env create -f env.yaml
conda activate gaslt
python -m signjoey train configs/train_pht.yaml --gpu_id 0
python -m signjoey test configs/train_pht.yaml --ckpt /path/to/checkpoint --gpu_id 0
```

For folds 2–9, repeat preprocessing with the desired fold number. Copy the training YAML and update `data.data_path`, `model.sim_name_to_video_id_json`, `model.sim_video_cos_sim`, and `training.model_dir` to use that fold. Use the same fold config for training and testing.

## Citation and credits

Please cite the original GASLT paper when using the model. If you use the signer-independent preprocessing, please also credit Keren Artiaga and link to this repository.

```bibtex
@inproceedings{yin2023gloss,
  title={Gloss attention for gloss-free sign language translation},
  author={Yin, Aoxiong and Zhong, Tianyun and Tang, Li and Jin, Weike and Jin, Tao and Zhao, Zhou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2551--2562},
  year={2023}
}
```

The upstream implementation builds on [SLT](https://github.com/neccam/slt), [WLASL](https://github.com/dxli94/WLASL), [SP-10](https://github.com/MLSLT/SP-10), [TSPNet](https://github.com/verashira/TSPNet), and [BPEmb](https://github.com/bheinzerling/bpemb).

# External baseline backbones

Four baselines are not obtainable from `timm`: **Conformer-S**, **Mobile-Former**,
**LSNet-T** and **MambaVision-T**. `adapters.py` provides a real factory for each,
and `verify_external_models.py` checks them without training.

| Registry key   | Model              | Source                                              | Input | Pretrained weights |
|----------------|--------------------|-----------------------------------------------------|-------|--------------------|
| `conformer`    | Conformer-S        | https://github.com/pengzhiliang/Conformer            | 224   | repo release       |
| `mobileformer` | Mobile-Former-294M | reference implementation (no official release)       | 224   | **usually none**   |
| `lsnet_t`      | LSNet-T            | https://github.com/jameslahm/lsnet                   | 224   | HF hub / release   |
| `mambavision` | MambaVision-T      | https://github.com/NVlabs/MambaVision                | 224   | pip / HF hub       |

## Install

```bash
mkdir -p checkpoints && export EXTERNAL_CKPT_DIR=$PWD/checkpoints

# Conformer -------------------------------------------------------------
git clone https://github.com/pengzhiliang/Conformer
export PYTHONPATH=$PWD/Conformer:$PYTHONPATH
# download Conformer_small_patch16.pth (link in that repo's README) -> checkpoints/

# LSNet -----------------------------------------------------------------
git clone https://github.com/jameslahm/lsnet
export PYTHONPATH=$PWD/lsnet:$PYTHONPATH
# importing the repo registers lsnet_t with timm; weights load from the HF hub

# MambaVision -----------------------------------------------------------
pip install mambavision          # needs mamba-ssm + causal-conv1d and a matching nvcc
#   or, with no CUDA kernels to build:
pip install "transformers>=4.40"  # -> nvidia/MambaVision-T-1K, trust_remote_code=True

# Mobile-Former ---------------------------------------------------------
git clone https://github.com/slwang9353/MobileFormer
export PYTHONPATH=$PWD/MobileFormer:$PYTHONPATH
```

Then wire the factories into the registry once:

```bash
python patch_models_registry.py --code-root /path/to/code     # idempotent
python verify_external_models.py --code-root /path/to/code --models conformer mobileformer lsnet_t mambavision
```

`verify_external_models.py` builds each model, counts its parameters, runs a
forward pass on `(2, 3, 256, 256)` and writes `results/model_provenance.json`.
The table generator reads that file, so the parameter column in the manuscript is
**measured, not quoted**; a model that was never verified prints `--`.

## Mobile-Former has no official ImageNet weights

Microsoft published the paper but not the code or the checkpoints. Every public
implementation is a reimplementation, and most ship no ImageNet weights. The
adapter therefore **refuses** to build it with `pretrained=True` unless you pass
`--allow-random-init`:

```bash
python verify_external_models.py --models mobileformer --allow-random-init
```

If you do that, `model_provenance.json` records `pretrained_loaded: false`, the
generated table prints `none (random)` in the pretraining column, and the
manuscript must not describe it as an ImageNet-pretrained baseline. A randomly
initialised 294M-parameter network compared against ImageNet-pretrained ones is
a different experiment, and a reviewer will notice.

## Details that matter

**Conformer returns two heads.** `Conformer.forward` returns
`[conv_logits, trans_logits]`; the official evaluation sums them. `MultiHeadSum`
does the same. Using only one head would be a different model from the published
one.

**No ImageNet normalisation anywhere.** This pipeline feeds `[0, 1]` tensors with
no mean/std normalisation to *every* model, including the six timm baselines
already reported. The new backbones are fed identically. Applying ImageNet
normalisation to only the new models would make them non-comparable with the
published rows; if you want normalisation, enable it for all models at once in
the dataloader and re-run everything.

**Head surgery is checked.** Checkpoints are loaded with `strict=False` after
dropping the 1000-class classifier tensors, and the loader then verifies that
every *missing* parameter is a classifier parameter. If anything else is missing
the checkpoint does not match the architecture and the build fails rather than
producing a partially initialised network.

**Failures are loud.** A backbone that is not installed raises
`ExternalModelUnavailable` with the install commands; `pretrained=True` that
cannot be honoured raises `PretrainedWeightsUnavailable`. Nothing is ever
silently substituted.

## Contract a factory must satisfy

Called as `factory(pretrained=<bool>, num_classes=2)`, returning an `nn.Module`
that

1. accepts float32 `(B, 3, 256, 256)` in `[0, 1]`, and
2. returns logits of shape `(B, 2)`.

Wrap a fixed-resolution backbone in `ResizeWrapper(model, size)` rather than
changing the shared dataloader — that is how CoAtNet (224) and CrossViT (240) are
already handled.

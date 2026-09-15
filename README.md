# BrNet major-revision setup

Start with [the setup report](reports/REVISION_SETUP_FINAL_REPORT.md). This repository was empty at intake; no prior code or experimental artifacts were overwritten. The protocol is a draft and no scientific training or test evaluation is authorized in this setup phase.

Run commands from the repository root with `.venv/Scripts/python.exe -m ...`. Scripts are modules. `scripts.reproduce_all` runs setup audits and tests only; it does not train. Public downloads are handled by `scripts.download_public`, whose official metadata inputs are archived. Raw data, virtual environments, predictions and checkpoints are excluded from Git.

The reference four-class BrNet and its explicit three-class adaptation are implemented. The full training, OOF, XAI and perturbation pipelines are not yet implemented or validated. Their scientific rules are prespecified in `configs/protocol_v2.yaml`; a draft configuration is not evidence that those experiments have run.

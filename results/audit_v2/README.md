# Additional audited outputs

These files were received after the initial final-artifact audit. Their numerical contents were independently reconciled against the row-level records on 25 September 2026.

## Evidence levels

- **External screened-subset evaluation — verified output, partial provenance.** The table contains 452 unique non-overlapping images, 233 correct predictions, and four-class probabilities consistent with the recorded predictions. The exact executed script and checkpoint file were not included; the execution log records checkpoint SHA-256 `193d7032f113f2ed6f0e4c1724003b44f3e6b887492d37c2b4e32d9ffa7b2a13`.
- **Matched-area perturbation — verified row-level output, partial provenance.** The 2,594-row table contains 86 unavailable matched controls and 2,508 analyzable pairs. Patient-clustered bootstrap estimates reproduce the reported results. Exact control-region coordinates, the exact executed script, and hashes for the 15 checkpoints were not retained in this package.
- **Grad-CAM parameter randomization — descriptive only.** The archive contains the 100-image sample manifest and 600 image-step records. Valid Spearman correlations vary from 50/100 to 92–95/100 because some maps are constant. The analysis uses one checkpoint and is not bit-identical across repeated GPU runs. No pass/fail claim is supported.
- **Compute benchmark — partial provenance.** Raw timings reproduce the summary for three CNNs. ViT-B/16 was not measured, concurrent GPU activity was not recorded, and one BrNet latency run was a large outlier. Median batch-1 latency is therefore the primary descriptive timing statistic. These measurements do not establish hardware-independent runtime superiority.

The exact scripts that generated these newly received outputs were not present in the supplied results archive. Repository scripts with similar names are historical implementations and must not be represented as byte-identical execution sources for these files.

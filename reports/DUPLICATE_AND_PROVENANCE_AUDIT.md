# Duplicate and provenance audit readiness

Exact raw and native-decoded hashes are implemented for Figshare. Available results are in results/duplicate_audit/exact_duplicates.csv. Native hashes retain shape and dtype, deliberately avoiding intensity quantization. Cross-format matching additionally needs a standardized grayscale conversion, pHash/dHash and embeddings; absence of an exact match cannot establish independence.

The completed exact-overlap pass covers all 4,201 recovered Figshare and Mendeley images. Machine-readable outcome: results/duplicate_audit/summary.json. Same raw filenames in different external class folders are recorded by ingestion and are not automatically duplicate images.

Full four-level cross-source analysis and human candidate review are **pending**, not passed. No cross-fold audit exists until approved duplicate decisions and patient manifests exist. No source identities are inferred from class labels.

Planned encoder: torchvision ResNet18 IMAGENET1K_V1, pooled penultimate features, official preprocessing, cosine similarity, immutable checkpoint checksum. Thresholds and review selection are in protocol_v2.yaml. Near-duplicate candidates require review; no automatic exclusions. Record reviewer, decision, rationale, pair IDs and whether the match crosses patient/source boundaries. Provenance remains unverified even if no duplicate is detected.

Aggregate constituent source reconstruction, source × class contingency, dimensions/intensity/border/compression confounding and source-held-out feasibility remain blocked by missing aggregate data. Treat source classification and any source-held-out experiment as exploratory.

The separately released Mendeley dataset is an **external-source dataset-shift stress test**. Publisher source claims and a certificate do not independently verify scanner/site/sequence provenance or patient independence.

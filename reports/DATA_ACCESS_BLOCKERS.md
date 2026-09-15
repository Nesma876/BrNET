# Data access status

## Kaggle aggregate — blocked

No KAGGLE_API_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY pair, ~/.kaggle/kaggle.json or ~/.kaggle/access_token was present at audit. Kaggle CLI was not installed. Do not send credentials in chat or store them in this repository.

In a separate access environment, install the official Kaggle CLI, record its resolved version, and use either the current API token mechanism (`KAGGLE_API_TOKEN` / the supported access_token location) or the legacy `~/.kaggle/kaggle.json` credentials generated in Kaggle account settings. Restrict credential-file permissions to your account. Then:

```powershell
kaggle datasets list -s "brain tumor"
kaggle datasets download -d rm1000/brain-tumor-mri-scans -p data/raw/aggregate
```

Record the actual downloaded version, license, byte size and SHA256 before ingestion. Do not claim the expected 7,023 count as observed. No aggregate split was recovered or created.

## Public sources

Figshare V8 metadata: https://api.figshare.com/v2/articles/1512427/versions/8

Mendeley V2 file listing: https://data.mendeley.com/public-api/datasets/82mtzd8x72/files?folder_id=root&version=2

Both official metadata responses are archived in data/manifests. The initial guessed Mendeley versions endpoint returned 404; the working file listing was subsequently found. A Figshare webpage fetch returned 403, but its official API is accessible. Interrupted/incomplete transfers are not valid archives and must never enter analyses. `scripts.download_public` checks publisher hashes and sizes.

Public sources need no HuggingFace credentials. Download registry entries represent successfully validated local files only. See data audit for exact completion status.

## Final recovery outcome

All six Figshare V8 files and the Mendeley V2 archive were recovered and validated. Their ingestion audits completed. Public-data access is no longer a blocker. An interrupted redundant `.part` transfer may remain ignored under raw data; it is not a dataset artifact. Kaggle remains blocked.

Official Kaggle authentication documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md . It also supports interactive `kaggle auth login`; this was not initiated in this setup.

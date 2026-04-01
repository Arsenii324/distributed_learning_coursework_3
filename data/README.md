# Dataset cache for distopt

This repository keeps downloaded datasets out of git. The cache directory is ignored via `.gitignore`.

## Where to put files

By default, `distopt` downloads/looks under:

- `data/libsvm/<name>`

You can override the base directory by setting:

- `DISTOPT_DATA_DIR=/absolute/path/to/cache`

Then the expected location becomes:

- `$DISTOPT_DATA_DIR/libsvm/<name>`

## Supported LIBSVM datasets (current demo)

The demo script expects the file name to match the dataset name:

- `a9a`  → `data/libsvm/a9a`
- `w8a`  → `data/libsvm/w8a`
- `SUSY` → `data/libsvm/SUSY`

## Download URLs

These are the URLs used by the downloader:

- a9a:  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
- w8a:  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a
- SUSY: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY

## If Python TLS verification fails

Some Python builds may fail TLS verification for that host. Options:

1) Manual download (browser or `curl`) and place the file at the expected path.

2) Use the demo flag `--insecure_download` (not recommended on untrusted networks).

Example:

- `python -m research.code.distopt.examples.run_libsvm_ridge --dataset a9a --insecure_download`

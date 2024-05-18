# HPC Tutorial
Collection of simple codes to initiate users in high performance computing (HPC) usage on a typical cluster

## Notes by Nishant, Aseem, Sanjit, Mayur

### Aseem 18/05/24: Installed [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) on HPC user `iucaa1`
procedure for pip installation of package named `package` without internet using suggestions from [this url](https://stackoverflow.com/questions/36725843/installing-python-packages-without-internet-and-using-source-code-as-tar-gz-and) and [this url](https://stackoverflow.com/questions/75514846/pip-says-version-40-8-0-of-setuptools-does-not-satisfy-requirement-of-setuptools):
1. On local machine having internet: `pip download package -d "/path/to/local/folder"`
2. Make tarball of /path/to/local/folder/ and transfer to remote HPC using `scp`
3. Decompress on remote HPC and navigate to the decompressed folder
4. `pip install package -f ./ --no-index --no-build-isolation`. The flag `--no-index` forces pip to search for all wheels in the current folder. The flag `--no-build-isolation` forces pip to look for dependency requirements locally.

Similarly installed `mpi4py` for user `iucaa1`.

### Nishant 17/05/24?: Added files in folder mhd_notes_pencil
contains notes, pencil-code examples, etc with a README

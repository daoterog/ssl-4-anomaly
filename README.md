# SSL-anomaly

## Setup

## Setup

To ensure a smooth development environment, we recommend using `conda` to manage
dependencies and virtual environments. (Follow the [Installing Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)
guide to install `conda` if you haven't already.)

### Step 1: Create a Virtual Environment

First, create a virtual environment using `conda`. Replace `env_name` with your
desired environment name:

```bash
conda create --name env_name python=3.10
```

Activate the environment:

```bash
conda activate env_name
```

### Step 2: Install Requirements

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

## Data

### Credentials Setup

To download the data, it is necessary to have the necessary environment
variables set. The following environment variables are required:

- `SEWERML_URL`
- `SEWERML_PASSWORD`

The `SEWERML_URL` is the URL to the data, and the `SEWERML_PASSWORD` is the
password to access the data. The data is hosted on a ScienceData website.
To obtain the credentials, register on [Sewer-ML Dataset Access Form](https://docs.google.com/forms/d/e/1FAIpQLSePeQXZqsuCFVacid92_arUBy2e42q4MzewqPkYqX3SrT-NPQ/viewform).

### Download

To download the data, run the following command:

```bash
bash data/download_script.sh
```

This shell script will download each of the files separately and save them in
the `data/sewer-ml` directory. Most of the files are compressed in `.zip`
format. If the download fails the script should be run again to resume the
download. The bash script will download the necessary drivers to start the
download process using `selenium`.

### Unzipping the data

After the download is complete, the data needs to be unzipped. To unzip the
data, run the following command:

```bash
bash data/unzip_downloaded_files.sh
```

This shell script will unzip all the files in the `data/sewer-ml` directory.
and save them in the `data/sewer-ml/images` directory. You can changes the input
and output directories in the script if needed within the script. After the
unzipping is complete, the `.zip` files will be removed from the directory for
space saving.

### Run supervised ablation

We will run supervised ablations for ResNet-18 and ViT-Tiny models, training them for
`45` epochs. The optimization settings are the following:

```yaml
- optimizer: adamw
- lr: 0.0005
- momentum: 0.9
- weight_decay: 0.0001
- exclude_bias_and_norm: false
- channels_last: true
- detach_backbone: false
- scheduler: cosine
- warmup_start_lr: 3.0e-05
- scheduler_interval: step
- warmup_epochs: 10
- eta_min: 1.0e-06
```

We will first run the ablations for the image resolution. This experiments will be ran
with no augmentations and will try `img_size=[64, 128, 256, 384, 512]`. For this you
should run the following command:

```bash
bash resolution_ablation.sh <batch_size>
```

Substitute `<batch_size>` with the desired batch size.

After the resolution ablation is complete, we will run the ablation for the augmentations.
The augmentations ablation is meant to be ran against the best performing resolution from
the resolution ablation. The augmentations will include `RandomResizedCrop` and
`ColorJitter` and the script ablates over the crop `min_scale=[0.08, 0.185, 0.29, 0.395, 0.5]`
and a `t_val=[0.1, 0.275, 0.45, 0.625, 0.8]` for the color jitter parameters. For this you
should run the following command:

```bash
bash augmentations_ablation.sh <img_size> <batch_size>
```

Substitute `<img_size>` with the desired resolution and `<batch_size>` with the desired
batch size.

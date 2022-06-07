# PostGIS with Plotly Dash to query and show lidar data

## Description

The repository contains code to show how to use python and [PostGIS](https://postgis.net/) for data analysis.

[Original data on the Sketchfab](https://skfb.ly/6QUwN)

## Requirements

1. Anaconda or Miniconda
3. [docker compose v2](https://github.com/docker/compose)

## How to run

Install dependencies:
```
conda env create -n env_name --file ./environment.yaml
conda activate env_name
```

Use [Mamba instead conda](https://github.com/mamba-org/mamba) to speedup dependency resolving.

Run PostGIS:
```
docker compose up -d
```

Check connection to the database.

Download data from the Sketchfab or any PLY file with color.

Modify config: [insert_data.yaml](configs/insert_data.yaml). By default, `drop_tables` is true.

Insert data:
```
python ./insert_data.py
```

Run dashboard:
```
python ./dashboard.py
```

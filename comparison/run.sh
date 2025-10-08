#!/bin/bash
# process_all.sh

echo "Starting benchmark analysis..."

python metrics_benchmark.py -d dlpfc_sample1 -m RAW
python metrics_benchmark.py -d dlpfc_sample1 -m PRECAST
python metrics_benchmark.py -d dlpfc_sample1 -m DeepST
python metrics_benchmark.py -d dlpfc_sample1 -m STAligner
python metrics_benchmark.py -d dlpfc_sample1 -m GraphST
python metrics_benchmark.py -d dlpfc_sample1 -m SPIRAL
python metrics_benchmark.py -d dlpfc_sample1 -m STitch3D
python metrics_benchmark.py -d dlpfc_sample1 -m Spatialign

python metrics_benchmark.py -d dlpfc_sample2 -m RAW
python metrics_benchmark.py -d dlpfc_sample2 -m PRECAST
python metrics_benchmark.py -d dlpfc_sample2 -m DeepST
python metrics_benchmark.py -d dlpfc_sample2 -m STAligner
python metrics_benchmark.py -d dlpfc_sample2 -m GraphST
python metrics_benchmark.py -d dlpfc_sample2 -m SPIRAL
python metrics_benchmark.py -d dlpfc_sample2 -m STitch3D
python metrics_benchmark.py -d dlpfc_sample2 -m Spatialign

python metrics_benchmark.py -d dlpfc_sample3 -m RAW
python metrics_benchmark.py -d dlpfc_sample3 -m PRECAST
# python metrics_benchmark.py -d dlpfc_sample3 -m DeepST
python metrics_benchmark.py -d dlpfc_sample3 -m STAligner
python metrics_benchmark.py -d dlpfc_sample3 -m GraphST
python metrics_benchmark.py -d dlpfc_sample3 -m SPIRAL
python metrics_benchmark.py -d dlpfc_sample3 -m STitch3D
python metrics_benchmark.py -d dlpfc_sample3 -m Spatialign

python metrics_benchmark.py -d dlpfc_7374 -m RAW
python metrics_benchmark.py -d dlpfc_7374 -m PRECAST
python metrics_benchmark.py -d dlpfc_7374 -m DeepST
python metrics_benchmark.py -d dlpfc_7374 -m STAligner
python metrics_benchmark.py -d dlpfc_7374 -m GraphST
python metrics_benchmark.py -d dlpfc_7374 -m SPIRAL
python metrics_benchmark.py -d dlpfc_7374 -m STitch3D
python metrics_benchmark.py -d dlpfc_7374 -m Spatialign

python metrics_benchmark.py -d dlpfc_all -m RAW
python metrics_benchmark.py -d dlpfc_all -m PRECAST
python metrics_benchmark.py -d dlpfc_all -m DeepST
python metrics_benchmark.py -d dlpfc_all -m STAligner
python metrics_benchmark.py -d dlpfc_all -m GraphST
python metrics_benchmark.py -d dlpfc_all -m SPIRAL
python metrics_benchmark.py -d dlpfc_all -m STitch3D
python metrics_benchmark.py -d dlpfc_all -m Spatialign

python metrics_benchmark.py -d hbc -m RAW
python metrics_benchmark.py -d hbc -m PRECAST
python metrics_benchmark.py -d hbc -m DeepST
python metrics_benchmark.py -d hbc -m STAligner
python metrics_benchmark.py -d hbc -m GraphST
python metrics_benchmark.py -d hbc -m SPIRAL
python metrics_benchmark.py -d hbc -m Spatialign

python metrics_benchmark.py -d coronal -m RAW
python metrics_benchmark.py -d coronal -m PRECAST
python metrics_benchmark.py -d coronal -m DeepST
python metrics_benchmark.py -d coronal -m STAligner
python metrics_benchmark.py -d coronal -m GraphST
python metrics_benchmark.py -d coronal -m SPIRAL
python metrics_benchmark.py -d coronal -m Spatialign

python metrics_benchmark.py -d mob -m RAW
python metrics_benchmark.py -d mob -m PRECAST
python metrics_benchmark.py -d mob -m STAligner
python metrics_benchmark.py -d mob -m GraphST
python metrics_benchmark.py -d mob -m SPIRAL
python metrics_benchmark.py -d mob -m Spatialign

echo "All datasets processed!"
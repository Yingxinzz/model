
echo "Starting benchmark analysis..."

# 处理所有DLPFC样本
python comparison_umap.py --modes 2
python comparison_umap.py --modes 3
python comparison_umap.py --modes 4
python comparison_umap.py --modes 5
python comparison_umap.py --modes 6
python comparison_umap.py --modes 7

echo "All datasets processed!"
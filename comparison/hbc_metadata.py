import scanpy as sc
import pandas as pd

meta_files = {
    "section1": "../RAW_SLICE/hbc/section1/meta.csv",
    "section2": "../RAW_SLICE/hbc/section2/meta.csv"
}

import pandas as pd
import os

for section, filepath in meta_files.items():
    df = pd.read_csv(filepath, index_col=0)
    
    df["celltype"] = df.idxmax(axis=1)
    
    df["batch"] = section
    
    df = df[["celltype", "batch"]]
    
    output_dir = os.path.dirname(filepath)
    output_csv = os.path.join(output_dir, f"metadata.csv")
    
    df.to_csv(output_csv)

meta_files = {
    "section1": "../RAW_SLICE/hbc/section1/metadata.csv",
    "section2": "../RAW_SLICE/hbc/section2/metadata.csv"
}

celltype_maps = {
    "section2": {
        "Cancer LumA SC": "Cancer", "Cancer LumB SC": "Cancer", "Cancer Cycling": "Cancer",
        "Plasmablasts": "Plasmablasts", "Myeloid_c2_LAM2_APOE": "Myeloid",
        "CAFs myCAF like s5": "CAFs", "Endothelial ACKR1": "Endothelial",
        "Myeloid_c9_Macrophage_2_CXCL10": "Myeloid", "T_cells_c11_MKI67": "T cells",
        "Endothelial CXCL12": "Endothelial", "Mature Luminal": "Luminal",
        "Cycling_Myeloid": "Myeloid", "Luminal Progenitors": "Luminal",
        "Cycling PVL": "PVL", "T_cells_c3_CD4+_Tfh_CXCL13": "T cells",
        "Cancer Basal SC": "Cancer", "B cells Naive": "B cells",
        "Endothelial RGS5": "Endothelial", "Endothelial Lymphatic LYVE1": "Endothelial",
        "Myeloid_c1_LAM1_FABP5": "Myeloid", "T_cells_c8_CD8+_LAG3": "T cells",
        "Myoepithelial": "Myoepithelial", "Myeloid_c0_DC_LAMP3": "Myeloid",
        "B cells Memory": "B cells", "CAFs MSC iCAF-like s1": "CAFs",
        "T_cells_c2_CD4+_T-regs_FOXP3": "T cells"
    },
    "section1": {
        "Cancer LumB SC": "Cancer", "Cancer LumA SC": "Cancer", "Plasmablasts": "Plasmablasts",
        "Cancer Cycling": "Cancer", "Mature Luminal": "Luminal", "CAFs myCAF like s5": "CAFs",
        "Myeloid_c2_LAM2_APOE": "Myeloid", "Endothelial ACKR1": "Endothelial",
        "Cancer Basal SC": "Cancer", "CAFs MSC iCAF-like s1": "CAFs",
        "T_cells_c11_MKI67": "T cells", "T_cells_c2_CD4+_T-regs_FOXP3": "T cells",
        "Endothelial CXCL12": "Endothelial", "Endothelial RGS5": "Endothelial",
        "CAFs Transitioning s3": "CAFs", "Myeloid_c9_Macrophage_2_CXCL10": "Myeloid",
        "PVL Differentiated s3": "PVL", "Luminal Progenitors": "Luminal",
        "Myoepithelial": "Myoepithelial", "Endothelial Lymphatic LYVE1": "Endothelial",
        "Cycling_Myeloid": "Myeloid", "T_cells_c6_IFIT1": "T cells",
        "B cells Memory": "B cells", "T_cells_c8_CD8+_LAG3": "T cells",
        "Myeloid_c1_LAM1_FABP5": "Myeloid", "T_cells_c0_CD4+_CCR7": "T cells",
        "B cells Naive": "B cells", "Cancer Her2 SC": "Cancer",
        "T_cells_c3_CD4+_Tfh_CXCL13": "T cells", "Myeloid_c4_DCs_pDC_IRF7": "Myeloid",
        "Cycling PVL": "PVL", "Myeloid_c0_DC_LAMP3": "Myeloid",
        "T_cells_c5_CD8+_GZMK": "T cells", "T_cells_c4_CD8+_ZFP36": "T cells",
        "Myeloid_c3_cDC1_CLEC9A": "Myeloid"
    }
}

def process_section(section, filepath, mapping, suffix):
    df = pd.read_csv(filepath, index_col=0)
    df['celltype'] = df['celltype'].map(mapping)
    df_coarse = df[['celltype', 'batch']]
    df_coarse.index = [idx.rsplit('-', 1)[0] + suffix for idx in df_coarse.index]
    print(df_coarse.index[:10])
    output_path = os.path.join(os.path.dirname(filepath), "metadata.csv")
    df_coarse.to_csv(output_path)
    print(df_coarse['celltype'].value_counts(), "\n")

for i, (sec, path) in enumerate(meta_files.items(), start=1):
    suffix = '-1' if sec == 'section1' else '-2'
    process_section(sec, path, celltype_maps[sec], suffix)

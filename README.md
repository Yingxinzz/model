# Towards a Better Understanding of Batch Effects in Spatial Transcriptomics: Definition and Method Evaluation


Spatial transcriptomics (ST) enables high-resolution profiling of gene expression within tissue slices, but technical and experimental variations introduce batch effects that obscure biological signals and complicate data integration. We present a framework that systematically defines and categorizes batch effects in ST into four types: (1) inter-slice, (2) Inter-sample, (3) cross-protocol/platform, and (4) Intra-slice. We further use it to evaluate the batch correction capability of representative integration methods across human and mouse datasets. Correction performance was quantified using ten metrics capturing both effective batch effects removal and the preservation of biological signals. The analysis reveals method-specific strengths, limitations, and characteristic trade-offs within batch correction. The framework and benchmark provide a practical evaluation reference for assessing and improving batch correction in spatial transcriptomic studies.



Datasets:

Dataset1 consists of 12 human DLPFC sections, available at https://research.libd.org/spatialLIBD/ with manual annotation

Dataset2 consists of Block A Section 1:  https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1; 
Block A Section 2:https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_2

Dataset3 consists of 3 slices from 3 different platforms: 
10X: https://www.10xgenomics.com/datasets/adult-mouse-olfactory-bulb-1-standards;
The processed Stereo-seq and Slide-seqV2 data can be downloaded from: https://drive.google.com/drive/folders/1Omte1adVFzyRDw7VloOAQYwtv_NjdWcG?usp=share_link.

Datasets4 consists of 3 slices from 3 different experimental protocols: "10Normal" https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-1-0;
"10X-DAPI":https://www.10xgenomics.com/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0;
"10X-FFPE":https://www.10xgenomics.com/datasets/adult-mouse-brain-ffpe-1-standard-1-3-0X.


Methods:

GraphST: https://github.com/JinmiaoChenLab/GraphST/tree/main

DeepST: https://github.com/JiangBioLab/DeepST

PRECAST:https://github.com/feiyoung/PRECAST

STAligner : https://github.com/zhanglabtools/STAligner

SPIRAL : https://github.com/guott15/SPIRAL

STitch3D : https://github.com/YangLabHKUST/STitch3D

spatiAlign : https://github.com/STOmics/Spatialign


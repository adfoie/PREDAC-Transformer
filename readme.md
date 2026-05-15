# Code of PREDAC-Transformer for: [Deciphering the Antigenic Evolution of Seasonal Influenza A Viruses with PREDAC-Transformer: From Antigenic Clustering to Key Site Identification]

This repository contains the Python scripts used for the analyses in the paper "[Deciphering the Antigenic Evolution of Seasonal Influenza A Viruses with PREDAC-Transformer: From Antigenic Clustering to Key Site Identification]". The project focuses on predicting the antigenic relationship of influenza viruses (H1N1 and H3N2) using a deep learning model called PREDAC-Transformer.

---

## Contents

The repository is organized into the following main directories:

*   `/csv`: Contains the primary input datasets in CSV format for testing (e.g., `inputdata_test.csv`).
*   `/script`: Contains all the Python scripts required to run the model training and analysis.
*   `/result`: Contains the prediction and clustering results.
*   `/model`: Contains the trained models for antigenic evolution.
*   `/seq`: Contains the sequence data for testing (e.g., `sequences_test.fasta`).
*   `/external`: Contains the AAIndex data (e.g., `aaindex_feature_H1N1.txt`).

### File and Folder Descriptions

```
.
├── csv/
│   ├── inputdata_test.csv     # Input data file for testing the model.
│
├── seq/
│   └── sequences_test.fasta      # Sequence file for testing the model.
│
├── external/
│   ├── aaindex_feature_H1N1.txt  # AAIndex feature data for H1N1 antigenic evolution.
│   └── aaindex_feature_H3N2.txt  # AAIndex feature data for H3N2 antigenic evolution.
│ 
├── script/
│   ├── generate_esm.py            # Script to generate ESM embeddings for sequences.
│   ├── generate_matrix.py         # Script to generate the input feature matrix.
│   ├── PredacTransformer_train.py # Main script to train the PREDAC-Transformer model.
│   ├── PredacTransformer_predict.py # Script to make predictions using the trained model.
│   └── kmeans_cluster.py          # Script for clustering the predicted antigenic sites.
│
├── result/
│   ├── pred/
│   │   ├── pred.csv              # Prediction results for antigenic evolution.
│   └── cluster/
│       ├── cluster_results.csv   # K-means clustering results of predicted antigenic sites.
│
└── model/
    ├── model_01/                 # Folder containing the trained model.
    └── model_02/                
```

---

## System Requirements & Dependencies

The code was developed and tested using Python 3.7.12. To run the scripts, you will need to install the following major libraries. We recommend using a virtual environment (e.g., `conda` or `venv`).


*   `tensorflow=2.7.0`
*   `torch=1.10.1`
*   `scikit-learn=0.19.2`
*   `pandas=1.3.5`
*   `numpy=1.21.6`
*   `keras=2.7.0`


---

## Usage Instructions

1.  **Navigate to the `script` directory:**
    ```bash
    cd script
    ```

2.  **Generate ESM embeddings:**
    To generate ESM embeddings for the sequence data, run the following command:
    ```bash
    python generate_esm.py \
      --seq /path/to/seq/sequences_test.fasta \
      --datatype H1N1_HA1 (or H3N2_HA1) \
      --outdir /path/to/output/directory
    ```
    Install ESM by following the instructions on the [Facebook ESM GitHub repository](https://github.com/facebookresearch/esm).

    After installing ESM, you can extract the ESM features using the following command:

    ```bash
    mkdir -p /path/to/H1N1_esm_seq (or H3N2_esm_seq)
    esm-extract esm2_t6_8M_UR50D /path/to/H1N1_HA1_forhi.fasta(or H3N2_HA1_forhi.fasta) /path/to/H1N1_esm_seq(or H3N2_esm_seq) --repr_layers 0 5 6 --include per_tok
    ```


3.  **Generate Input Matrix:**
    After generating ESM embeddings, run the following to generate the input feature matrix:
    ```bash
    python generate_matrix.py \
      --aaindex_file /path/to/external/aaindex_feature_H1N1.txt (or aaindex_feature_H3N2.txt) \
      --esm_dir /path/to/esm/embeddings \
      --seq_dir /path/to/csv \
      --dir /path/to/csv \
      --thread 50
    ```

4.  **Train the Model:**
    To train the PREDAC-Transformer model, run the following:
    ```bash
    python PredacTransformer_train.py \
      --input_dir /path/to/csv \
      --filename inputdata_test.csv.npy \
      --subtype H1N1 (or H3N2) \
      --number_columns 654 --epoch 50
    ```

5.  **Make Predictions:**
    After training, you can use the model to make predictions on new data:
    ```bash
    python PredacTransformer_predict.py \
      --input_dir /path/to/csv \
      --filename inputdata_test.csv.npy \
      --csv_file inputdata_test.csv \
      --number_columns 654 \
      --subtype H1N1 (or H3N2) \
      --outdir /path/to/result/pred \
      --model_dir /path/to/model
    ```

6.  **Cluster Results:**
    Finally, you can perform k-means clustering on the prediction results:
    ```bash
    python kmeans_cluster.py \
      --inputcsv /path/to/result/pred/pred.csv \
      --outdir /path/to/result/cluster \
      --subtype H1N1 (or H3N2)
    ```

---

## License

*   The code in the `/script` directory is released under the **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.

---

## Contact & Citation

For any questions regarding the code or data, please contact [Jingze Liu] at [liujz1211@163.com].

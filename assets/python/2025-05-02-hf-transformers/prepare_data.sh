#!/bin/bash

mkdir -p data/train
mkdir -p data/test

urls=(
    "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz"
    "https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz"
    "https://ftp.ensembl.org/pub/release-113/fasta/drosophila_melanogaster/cds/Drosophila_melanogaster.BDGP6.46.cds.all.fa.gz"
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-60/fasta/bacteria_12_collection/escherichia_coli_gca_001606525/cds/Escherichia_coli_gca_001606525.ASM160652v1.cds.all.fa.gz"
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/release-60/fasta/saccharomyces_cerevisiae/cds/Saccharomyces_cerevisiae.R64-1-1.cds.all.fa.gz"
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/cds/Arabidopsis_thaliana.TAIR10.cds.all.fa.gz"
)

for url in "${urls[@]}"; do

    species=$(basename "$url" | cut -d. -f1)
    echo "Processing ${species}..."

    curl -o tmp.fa.gz "$url"
    echo "Downloaded ${species} data."

    gunzip tmp.fa.gz

    # remove headers and shorter sequences
    grep -v ">" tmp.fa | awk 'length($0) == 60' | shuf >shuf.fa
    echo "Processed ${species} data."

    head -n 1000 shuf.fa > data/train/${species}.txt
    tail -n 1000 shuf.fa > data/test/${species}.txt
    echo "Split ${species} data into train and test sets."

    rm tmp.fa shuf.fa
    echo "Cleaned up temporary files."
done

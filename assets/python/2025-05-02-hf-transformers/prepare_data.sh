#!/bin/bash
set -o errexit
set -o pipefail

mkdir -p data/train
mkdir -p data/test

N_SAMPLES=1000
URLS=(
    "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz"
    "https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz"
    "https://ftp.ensembl.org/pub/release-113/fasta/drosophila_melanogaster/cds/Drosophila_melanogaster.BDGP6.46.cds.all.fa.gz"
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-60/fasta/bacteria_12_collection/escherichia_coli_gca_001606525/cds/Escherichia_coli_gca_001606525.ASM160652v1.cds.all.fa.gz"
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/release-60/fasta/saccharomyces_cerevisiae/cds/Saccharomyces_cerevisiae.R64-1-1.cds.all.fa.gz"
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/cds/Arabidopsis_thaliana.TAIR10.cds.all.fa.gz"
)

for URL in "${URLS[@]}"; do

    SPECIES=$(basename "${URL}" | cut -d. -f1)
    echo "Processing ${SPECIES}"

    echo -e "\tDownloading coding DNA sequences..."
    curl -s -o cds.fa.gz "${URL}" && gunzip cds.fa.gz

    echo -e "\tRemoving headers and shorter sequences..."
    grep -v ">" cds.fa | awk 'length($0) == 60' | shuf >filt_shuff_cds.fa

    echo -e "\tCreating train and test sets, with ${N_SAMPLES} sequences each..."
    head -n $N_SAMPLES filt_shuff_cds.fa > data/train/"${SPECIES}".txt
    tail -n $N_SAMPLES filt_shuff_cds.fa > data/test/"${SPECIES}".txt

    echo -e "\tRemoving temporary files..."
    rm cds.fa filt_shuff_cds.fa

done

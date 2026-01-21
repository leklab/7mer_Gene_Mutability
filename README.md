# Mutation Rate Table for Coding Genes (7-mer Context)

## Overview

This repository contains code to generate a **genome-wide mutation rate table for all human protein-coding genes**, based on **7-mer nucleotide mutation rates** derived from:

> Carlson et al., *Extremely rare variants reveal patterns of germline mutation rate heterogeneity in humans*  
> **Nature Communications (2018)**  
> https://www.nature.com/articles/s41467-018-05936-5

The resulting mutation rate tables are stratified by **variant class** (e.g. synonymous, missense, nonsense, splice-site) and are intended for downstream analyses such as **gene burden testing**.

This pipeline was developed for analysis of sequencing data from the **Pediatric Cardiac Genomics Consortium (PCGC)**, but is broadly applicable to rare variant studies in other disease cohorts.

---

## Tools and Reference Versions

This project uses the following tools and versioned resources:

- **Python**
- **pyensembl** – Ensembl gene and transcript annotations  
  - Ensembl **Release 85** (human)
- **pysam** – FASTA-based reference genome access
- **Hail** – scalable variant annotation and aggregation
- **Ensembl VEP** – functional consequence annotations
- **Reference genome**: **GRCh38 / hg38** (matched to Ensembl 85)

---

## High-Level Pipeline Logic

The notebook implements the following steps:

1. Load Ensembl gene models and identify **protein-coding genes**
2. Select the **longest CDS per gene**
3. Extract **coding DNA sequences** from the hg38 reference genome
4. Generate **all possible single-nucleotide substitutions** in coding regions
5. Annotate variants with **functional consequence classes**
6. Assign **7-mer sequence context mutation rates**
7. Aggregate mutation rates by **gene and variant class**
8. Output per-gene mutation rate tables for downstream burden analyses

---

## Intended Use

The resulting mutation rate tables can be used to:

- Normalize rare variant counts in **gene burden analyses**
- Compare observed vs. expected mutation rates
- Stratify analyses by **functional variant class**
- Support statistical models in rare disease genomics

---

## Development

Developed and tested by **Kenneth Ng (Yale University)** on the **McCleary high-performance computing cluster**.

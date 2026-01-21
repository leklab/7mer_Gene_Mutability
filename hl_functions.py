##hl_functions.py
#code from Monkol, Nick, Amar, Spencer, etc.

import os
import hail as hl
# import gnomad
import pprint
from typing import *
from scipy import stats
import numpy as np
import pandas as pd

# Consequence terms in order of severity (more severe to less severe) as estimated by Ensembl.
# See https://ensembl.org/info/genome/variation/prediction/predicted_data.html
CONSEQUENCE_TERMS = [
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",  # new in v81
    "initiator_codon_variant",  # deprecated
    "transcript_amplification",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",  # new in v79
    "splice_region_variant",
    "incomplete_terminal_codon_variant",
    "start_retained_variant",
    "stop_retained_variant",
    "synonymous_variant",
    "coding_sequence_variant",
    "mature_miRNA_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "non_coding_exon_variant",  # deprecated
    "intron_variant",
    "NMD_transcript_variant",
    "non_coding_transcript_variant",
    "nc_transcript_variant",  # deprecated
    "upstream_gene_variant",
    "downstream_gene_variant",
    "TFBS_ablation",
    "TFBS_amplification",
    "TF_binding_site_variant",
    "regulatory_region_ablation",
    "regulatory_region_amplification",
    "feature_elongation",
    "regulatory_region_variant",
    "feature_truncation",
    "intergenic_variant",
]

# hail DictExpression that maps each CONSEQUENCE_TERM to it's rank in the list
CONSEQUENCE_TERM_RANK_LOOKUP = hl.dict({term: rank for rank, term in enumerate(CONSEQUENCE_TERMS)})


OMIT_CONSEQUENCE_TERMS = [
    "upstream_gene_variant",
    "downstream_gene_variant",
]

def annotate_fisher_helper(
    a: hl.expr.Int64Expression,
    b: hl.expr.Int64Expression,
    c: hl.expr.Int64Expression,
    d: hl.expr.Int64Expression
) -> hl.expr.Int64Expression:
    """
    """
    return (
            (
             hl.fisher_exact_test(hl.int(a), hl.int(b), hl.int(c), hl.int(d))
            )
    )

def annotate_fisher_locus(
        mt: hl.MatrixTable,
) -> hl.MatrixTable:
    """
    Annotate with fisher, locus
    """
    return mt.annotate_rows(fisher=annotate_fisher_helper(mt.gt_countS, mt.gt_countnotS, mt.gt_countN, mt.gt_countnotN))

def annotate_fisher_gene(
        mt: hl.MatrixTable,
) -> hl.MatrixTable:
    """
    Annotate with fisher, gene level
    """
    return mt.annotate_rows(fisher=annotate_fisher_helper(mt.genecountS, mt.genecountnotS, mt.genecountN, mt.genecountnotN))


def get_het_homo_hemi_spec(
    gt_expr: hl.expr.CallExpression
) -> hl.expr.Int32Expression:
    """
    """
    return (
            (
                hl.case()
                .when(gt_expr.is_haploid(),31)
                .when(gt_expr.is_het_non_ref(),11)
                .when(gt_expr.is_het_ref(),12)
                .when(gt_expr.is_hom_var(), 21)
                .when(gt_expr.is_hom_ref(), 22)
                .default(0)
            )
    )

def get_het_homo_hemi(
    gt_expr: hl.expr.CallExpression
) -> hl.expr.Int32Expression:
    """
    """
    return (
            (
                hl.case()
                .when(gt_expr.is_het(), 1)
                .when((gt_expr.is_hom_var())|(gt_expr.is_hom_ref()), 2)
                .when(gt_expr.is_haploid(), 3)
                .default(0)
            )
    )

def annotate_het_homo_hemi(
        mt: hl.MatrixTable,
) -> hl.MatrixTable:
    """
    Annotate with is_het
    """
    return mt.annotate_entries(het_homo_hemi=get_het_homo_hemi(mt.GT),het_homo_hemi_spec=get_het_homo_hemi_spec(mt.GT))


def get_adj_expr(
        gt_expr: hl.expr.CallExpression,
        gq_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
        dp_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
        ad_expr: hl.expr.ArrayNumericExpression,
        adj_gq: int = 20,
        adj_dp: int = 8,
        adj_ab: float = 0.25,
        haploid_adj_dp: int = 10
) -> hl.expr.BooleanExpression:
    """
    Gets adj genotype annotation.
    Defaults correspond to gnomAD values.
    """
    return (
            (gq_expr >= adj_gq) &
            hl.cond(
                gt_expr.is_haploid(),
                dp_expr >= haploid_adj_dp,
                dp_expr >= adj_dp
            ) &
            (
                hl.case()
                .when(~gt_expr.is_het(), True)
                .when(gt_expr.is_het_ref(), ad_expr[1] / dp_expr >= adj_ab)
                .default((ad_expr[0] / dp_expr >= adj_ab ) & (ad_expr[1] / dp_expr >= adj_ab ))
            )
    )

def annotate_adj(
        mt: hl.MatrixTable,
        adj_gq: int = 20,
        adj_dp: int = 8,
        adj_ab: float = 0.25,
        haploid_adj_dp: int = 10
) -> hl.MatrixTable:
    """
    Annotate genotypes with adj criteria (assumes diploid)
    Defaults correspond to gnomAD values.
    """
    return mt.annotate_entries(adj=get_adj_expr(mt.GT, mt.GQ, mt.DP, mt.AD, adj_gq, adj_dp, adj_ab, haploid_adj_dp))


def add_variant_type(alt_alleles: hl.expr.ArrayExpression) -> hl.expr.StructExpression:
    """                                                                                               
    Get Struct of variant_type and n_alt_alleles from ArrayExpression of Strings (all alleles)        
    """
    ref = alt_alleles[0]
    alts = alt_alleles[1:]
    non_star_alleles = hl.filter(lambda a: a != '*', alts)
    return hl.struct(variant_type=hl.cond(
        hl.all(lambda a: hl.is_snp(ref, a), non_star_alleles),
        hl.cond(hl.len(non_star_alleles) > 1, "multi-snv", "snv"),
        hl.cond(
            hl.all(lambda a: hl.is_indel(ref, a), non_star_alleles),
            hl.cond(hl.len(non_star_alleles) > 1, "multi-indel", "indel"),
            "mixed")
    ), n_alt_alleles=hl.len(non_star_alleles))

def generate_split_alleles(mt: hl.MatrixTable) -> hl.Table:

    allele_data = hl.struct(nonsplit_alleles=mt.alleles,
                            has_star=hl.any(lambda a: a == '*', mt.alleles))

    mt = mt.annotate_rows(allele_data=allele_data.annotate(**add_variant_type(mt.alleles)))
    mt = hl.split_multi_hts(mt,left_aligned=True)

    allele_type = (hl.case()
                   .when(hl.is_snp(mt.alleles[0], mt.alleles[1]), 'snv')
                   .when(hl.is_insertion(mt.alleles[0], mt.alleles[1]), 'ins')
                   .when(hl.is_deletion(mt.alleles[0], mt.alleles[1]), 'del')
                   .default('complex')
                   )
    mt = mt.annotate_rows(allele_data=mt.allele_data.annotate(allele_type=allele_type,
                                                              was_mixed=mt.allele_data.variant_type == 'mixed'))
    return mt

def get_expr_for_vep_consequence_terms_set(vep_transcript_consequences_root):
    return hl.set(vep_transcript_consequences_root.flatmap(lambda c: c.consequence_terms))


def get_expr_for_vep_gene_ids_set(vep_transcript_consequences_root, only_coding_genes=False):
    """Expression to compute the set of gene ids in VEP annotations for this variant.

    Args:
        vep_transcript_consequences_root (ArrayExpression): VEP transcript_consequences root in the struct
        only_coding_genes (bool): If set to True, non-coding genes will be excluded.
    Return:
        SetExpression: expression
    """

    expr = vep_transcript_consequences_root

    if only_coding_genes:
        expr = expr.filter(lambda c: hl.or_else(c.biotype, "") == "protein_coding")

    return hl.set(expr.map(lambda c: c.gene_id))


def get_expr_for_vep_protein_domains_set(vep_transcript_consequences_root):
    return hl.set(
        vep_transcript_consequences_root.flatmap(lambda c: c.domains.map(lambda domain: domain.db + ":" + domain.name))
    )

def get_expr_for_formatted_hgvs(csq):
    return hl.cond(
        hl.is_missing(csq.hgvsp) | HGVSC_CONSEQUENCES.contains(csq.major_consequence),
        csq.hgvsc.split(":")[-1],
        hl.cond(
            csq.hgvsp.contains("=") | csq.hgvsp.contains("%3D"),
            hl.bind(
                lambda protein_letters: "p." + protein_letters + hl.str(csq.protein_start) + protein_letters,
                PROTEIN_LETTERS_1TO3.get(csq.amino_acids),
                #hl.delimit(csq.amino_acids.split("").map(lambda l: PROTEIN_LETTERS_1TO3.get(l)), ""),
                #FIX THIS. somehow this is not parsing correctly
            ),
            csq.hgvsp.split(":")[-1],
        ),
    )


def get_expr_for_vep_sorted_transcript_consequences_array(vep_root,
                                                          include_coding_annotations=True,
                                                          omit_consequences=OMIT_CONSEQUENCE_TERMS):
    """Sort transcripts by 3 properties:

        1. coding > non-coding
        2. transcript consequence severity
        3. canonical > non-canonical

    so that the 1st array entry will be for the coding, most-severe, canonical transcript (assuming
    one exists).

    Also, for each transcript in the array, computes these additional fields:
        domains: converts Array[Struct] to string of comma-separated domain names
        hgvs: set to hgvsp is it exists, or else hgvsc. formats hgvsp for synonymous variants.
        major_consequence: set to most severe consequence for that transcript (
            VEP sometimes provides multiple consequences for a single transcript)
        major_consequence_rank: major_consequence rank based on VEP SO ontology (most severe = 1)
            (see http://www.ensembl.org/info/genome/variation/predicted_data.html)
        category: set to one of: "lof", "missense", "synonymous", "other" based on the value of major_consequence.

    Args:
        vep_root (StructExpression): root path of the VEP struct in the MT
        include_coding_annotations (bool): if True, fields relevant to protein-coding variants will be included
    """

    selected_annotations = [
        "biotype",
        "canonical",
        "cdna_start",
        "cdna_end",
        "codons",
        "gene_id",
        "gene_symbol",
        "hgvsc",
        "hgvsp",
        "transcript_id",
    ]

    if include_coding_annotations:
        selected_annotations.extend(
            [
                "amino_acids",
                "lof",
                "lof_filter",
                "lof_flags",
                "lof_info",
                "polyphen_prediction",
                "protein_id",
                "protein_start",
                "sift_prediction",
            ]
        )

    omit_consequence_terms = hl.set(omit_consequences) if omit_consequences else hl.empty_set(hl.tstr)

    result = hl.sorted(
        vep_root.transcript_consequences.map(
            lambda c: c.select(
                *selected_annotations,
                consequence_terms=c.consequence_terms.filter(lambda t: ~omit_consequence_terms.contains(t)),
                domains=c.domains.map(lambda domain: domain.db + ":" + domain.name),
                major_consequence=hl.cond(
                    c.consequence_terms.size() > 0,
                    hl.sorted(c.consequence_terms, key=lambda t: CONSEQUENCE_TERM_RANK_LOOKUP.get(t))[0],
                    hl.null(hl.tstr),
                )
            )
        )
        .filter(lambda c: c.consequence_terms.size() > 0)
        .map(
            lambda c: c.annotate(
                category=(
                    hl.case()
                    .when(
                        CONSEQUENCE_TERM_RANK_LOOKUP.get(c.major_consequence)
                        <= CONSEQUENCE_TERM_RANK_LOOKUP.get("frameshift_variant"),
                        "lof",
                    )
                    .when(
                        CONSEQUENCE_TERM_RANK_LOOKUP.get(c.major_consequence)
                        <= CONSEQUENCE_TERM_RANK_LOOKUP.get("missense_variant"),
                        "missense",
                    )
                    .when(
                        CONSEQUENCE_TERM_RANK_LOOKUP.get(c.major_consequence)
                        <= CONSEQUENCE_TERM_RANK_LOOKUP.get("synonymous_variant"),
                        "synonymous",
                    )
                    .default("other")
                ),
                hgvs=get_expr_for_formatted_hgvs(c),
                major_consequence_rank=CONSEQUENCE_TERM_RANK_LOOKUP.get(c.major_consequence),
            )
        ),
        lambda c: (
            hl.bind(
                lambda is_coding, is_most_severe, is_canonical: (
                    hl.cond(
                        is_coding,
                        hl.cond(is_most_severe, hl.cond(is_canonical, 1, 2), hl.cond(is_canonical, 3, 4)),
                        hl.cond(is_most_severe, hl.cond(is_canonical, 5, 6), hl.cond(is_canonical, 7, 8)),
                    )
                ),
                hl.or_else(c.biotype, "") == "protein_coding",
                hl.set(c.consequence_terms).contains(vep_root.most_severe_consequence),
                hl.or_else(c.canonical, 0) == 1,
            )
        ),
    )

    if not include_coding_annotations:
        # for non-coding variants, drop fields here that are hard to exclude in the above code
        result = result.map(lambda c: c.drop("domains", "hgvsp"))

    return hl.zip_with_index(result).map(
        lambda csq_with_index: csq_with_index[1].annotate(transcript_rank=csq_with_index[0])
    )


def get_expr_for_vep_protein_domains_set_from_sorted(vep_sorted_transcript_consequences_root):
    return hl.set(
        vep_sorted_transcript_consequences_root.flatmap(lambda c: c.domains)
    )


def get_expr_for_vep_gene_id_to_consequence_map(vep_sorted_transcript_consequences_root, gene_ids):
    # Manually build string because hl.json encodes a dictionary as [{ key: ..., value: ... }, ...]
    return (
        "{"
        + hl.delimit(
            gene_ids.map(
                lambda gene_id: hl.bind(
                    lambda worst_consequence_in_gene: '"' + gene_id + '":"' + worst_consequence_in_gene.major_consequence + '"',
                    vep_sorted_transcript_consequences_root.find(lambda c: c.gene_id == gene_id)
                )
            )
        )
        + "}"
    )


def get_expr_for_vep_transcript_id_to_consequence_map(vep_transcript_consequences_root):
    # Manually build string because hl.json encodes a dictionary as [{ key: ..., value: ... }, ...]
    return (
        "{"
        + hl.delimit(
            vep_transcript_consequences_root.map(lambda c: '"' + c.transcript_id + '": "' + c.major_consequence + '"')
        )
        + "}"
    )


def get_expr_for_vep_transcript_ids_set(vep_transcript_consequences_root):
    return hl.set(vep_transcript_consequences_root.map(lambda c: c.transcript_id))


def get_expr_for_worst_transcript_consequence_annotations_struct(
    vep_sorted_transcript_consequences_root, include_coding_annotations=True
):
    """Retrieves the top-ranked transcript annotation based on the ranking computed by
    get_expr_for_vep_sorted_transcript_consequences_array(..)

    Args:
        vep_sorted_transcript_consequences_root (ArrayExpression):
        include_coding_annotations (bool):
    """

    transcript_consequences = {
        "biotype": hl.tstr,
        "canonical": hl.tint,
        "category": hl.tstr,
        "cdna_start": hl.tint,
        "cdna_end": hl.tint,
        "codons": hl.tstr,
        "gene_id": hl.tstr,
        "gene_symbol": hl.tstr,
        "hgvs": hl.tstr,
        "hgvsc": hl.tstr,
        "major_consequence": hl.tstr,
        "major_consequence_rank": hl.tint,
        "transcript_id": hl.tstr,
    }

    if include_coding_annotations:
        transcript_consequences.update(
            {
                "amino_acids": hl.tstr,
                "domains": hl.tstr,
                "hgvsp": hl.tstr,
                "lof": hl.tstr,
                "lof_flags": hl.tstr,
                "lof_filter": hl.tstr,
                "lof_info": hl.tstr,
                "polyphen_prediction": hl.tstr,
                "protein_id": hl.tstr,
                "sift_prediction": hl.tstr,
            }
        )

    return hl.cond(
        vep_sorted_transcript_consequences_root.size() == 0,
        hl.struct(**{field: hl.null(field_type) for field, field_type in transcript_consequences.items()}),
        hl.bind(
            lambda worst_transcript_consequence: (
                worst_transcript_consequence.annotate(
                    domains=hl.delimit(hl.set(worst_transcript_consequence.domains))
                ).select(*transcript_consequences.keys())
            ),
            vep_sorted_transcript_consequences_root[0],
        ),
    )


def filter_major_consequence(de_novo_results):
        '''filter de novo calls based on their major consequence that are not in coding regions or splice sites'''
        
        de_novo_results_filtered = de_novo_results.filter(de_novo_results.major_consequence == "intron_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())

        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "3_prime_UTR_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "5_prime_UTR_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "splice_region_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "upstream_gene_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "downstream_gene_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "transcript_ablation", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "transcript_amplification", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "start_retained_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "stop_retained_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "intergenic_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "non_coding_transcript_exon_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "start_lost", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "regulatory_region_amplification", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "feature_elongation", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "regulatory_region_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        de_novo_results_filtered = de_novo_results_filtered.filter(de_novo_results_filtered.major_consequence == "TF_binding_site_variant", keep=False)
        pprint.pprint(de_novo_results_filtered.count())


        return de_novo_results_filtered

def filter_DP(de_novo_results):
        '''filter de novo results based on DP for parents and proband'''
        
        de_novo_results = de_novo_results.filter(de_novo_results.proband_entry.DP >= 10, keep=True)
        pprint.pprint(de_novo_results.count())
        
        de_novo_results = de_novo_results.filter(de_novo_results.father_entry.DP >= 10, keep=True)
        pprint.pprint(de_novo_results.count())
        
        de_novo_results = de_novo_results.filter(de_novo_results.mother_entry.DP >= 10, keep=True)
        pprint.pprint(de_novo_results.count())
        
        return de_novo_results 
   
   
     
# Filter matrix table before calling de novo events

# Filter out variants with a gnomad_af > 0.0004 
# mt = mt.filter_rows(mt.gnomad_af > 0.0004, keep=False) 

def filter_mt_gnomad_af(mt): 
        mt = mt.filter_rows(mt.gnomad_af > 0.0004, keep=False)
        pprint.pprint(mt.count())
        return mt


def filter_mt_major_consequence(mt):
        mt = mt.filter_rows(mt.major_consequence == "intron_variant", keep=False)
        pprint.pprint(mt.count())
        
        mt = mt.filter_rows(mt.major_consequence == "3_prime_UTR_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "5_prime_UTR_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "splice_region_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "upstream_gene_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "downstream_gene_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "transcript_ablation", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "transcript_amplification", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "start_retained_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "stop_retained_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "intergenic_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "non_coding_transcript_exon_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "start_lost", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "regulatory_region_amplification", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "feature_elongation", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "regulatory_region_variant", keep=False)
        #pprint.pprint(mt.count())


        mt = mt.filter_rows(mt.major_consequence == "TF_binding_site_variant", keep=False)
        pprint.pprint(mt.count())
        
        return mt
    
def generate_sample_lists(filename):
    samples = []
    with open(filename, "r") as cohort_file:
        for line in cohort_file:
            stripped_line = line.strip()
            samples.append(stripped_line)
    return samples

from hail.genetics.pedigree import Pedigree
from hail.matrixtable import MatrixTable
from hail.expr import expr_float64
from hail.table import Table
from hail.typecheck import typecheck, numeric
from hail.methods.misc import require_biallelic

@typecheck(mt=MatrixTable,
           pedigree=Pedigree,
           pop_frequency_prior=expr_float64,
           min_gq=int,
           min_p=numeric,
           max_parent_ab=numeric,
           min_child_ab=numeric,
           min_dp_ratio=numeric,
           ignore_in_sample_allele_frequency=bool)
def my_de_novo(mt: MatrixTable,
            pedigree: Pedigree,
            pop_frequency_prior,
            *,
            min_gq: int = 20,
            min_p: float = 0.05,
            max_parent_ab: float = 0.05,
            min_child_ab: float = 0.20,
            min_dp_ratio: float = 0.10,
            ignore_in_sample_allele_frequency: bool = False) -> Table:
  
    DE_NOVO_PRIOR = 1 / 30000000
    MIN_POP_PRIOR = 100 / 30000000

    required_entry_fields = {'GT', 'AD', 'DP', 'GQ', 'PL'}
    missing_fields = required_entry_fields - set(mt.entry)
    if missing_fields:
        raise ValueError(f"'de_novo': expected 'MatrixTable' to have at least {required_entry_fields}, "
                         f"missing {missing_fields}")

    pop_frequency_prior = hl.case() \
        .when((pop_frequency_prior >= 0) & (pop_frequency_prior <= 1), pop_frequency_prior) \
        .or_error(hl.str("de_novo: expect 0 <= pop_frequency_prior <= 1, found " + hl.str(pop_frequency_prior)))

    if ignore_in_sample_allele_frequency:
        # this mode is used when families larger than a single trio are observed, in which
        # an allele might be de novo in a parent and transmitted to a child in the dataset.
        # The original model does not handle this case correctly, and so this experimental
        # mode can be used to treat each trio as if it were the only one in the dataset.
        mt = mt.annotate_rows(__prior=pop_frequency_prior,
                              __alt_alleles=hl.int64(1),
                              __site_freq=hl.max(pop_frequency_prior, MIN_POP_PRIOR))
    else:
        n_alt_alleles = hl.agg.sum(mt.GT.n_alt_alleles())
        total_alleles = 2 * hl.agg.sum(hl.is_defined(mt.GT))
        # subtract 1 from __alt_alleles to correct for the observed genotype
        mt = mt.annotate_rows(__prior=pop_frequency_prior,
                              __alt_alleles=n_alt_alleles,
                              __site_freq=hl.max((n_alt_alleles - 1) / total_alleles,
                                                 pop_frequency_prior,
                                                 MIN_POP_PRIOR))

    mt = require_biallelic(mt, 'de_novo')

    tm = hl.trio_matrix(mt, pedigree, complete_trios=True)
    tm = tm.annotate_rows(__autosomal=tm.locus.in_autosome_or_par(),
                          __x_nonpar=tm.locus.in_x_nonpar(),
                          __y_nonpar=tm.locus.in_y_nonpar(),
                          __mito=tm.locus.in_mito())
    
    autosomal = tm.__autosomal | (tm.__x_nonpar & tm.is_female)
    hemi_x = tm.__x_nonpar & ~tm.is_female
    hemi_y = tm.__y_nonpar & ~tm.is_female
    hemi_mt = tm.__mito

    is_snp = hl.is_snp(tm.alleles[0], tm.alleles[1])
    n_alt_alleles = tm.__alt_alleles
    prior = tm.__site_freq
    
    # Updated for hemizygous child calls to not require a call in uninvolved parent
    has_candidate_gt_configuration = (
        ( autosomal & tm.proband_entry.GT.is_het() & 
          tm.father_entry.GT.is_hom_ref() & tm.mother_entry.GT.is_hom_ref() ) |
        ( hemi_x & tm.proband_entry.GT.is_hom_var() & tm.mother_entry.GT.is_hom_ref() ) |
        ( hemi_y & tm.proband_entry.GT.is_hom_var() & tm.father_entry.GT.is_hom_ref() ) |
        ( hemi_mt & tm.proband_entry.GT.is_hom_var() & tm.mother_entry.GT.is_hom_ref() ) )
    
    failure = hl.missing(hl.tstruct(p_de_novo=hl.tfloat64, confidence=hl.tstr))
    
    kid = tm.proband_entry
    dad = tm.father_entry
    mom = tm.mother_entry

    kid_linear_pl = 10 ** (-kid.PL / 10)
    kid_pp = hl.bind(lambda x: x / hl.sum(x), kid_linear_pl)

    dad_linear_pl = 10 ** (-dad.PL / 10)
    dad_pp = hl.bind(lambda x: x / hl.sum(x), dad_linear_pl)

    mom_linear_pl = 10 ** (-mom.PL / 10)
    mom_pp = hl.bind(lambda x: x / hl.sum(x), mom_linear_pl)

    kid_ad_ratio = kid.AD[1] / hl.sum(kid.AD)
    
    # Try to get these all to an expected value of 0.5
    dp_ratio = (hl.case()
                  .when(hemi_x, kid.DP / mom.DP) # Because mom is diploid but kid is not
                  .when(hemi_y, kid.DP / (2*dad.DP))
                  .when(hemi_mt, kid.DP / (2*mom.DP))
                  .when(tm.__x_nonpar & tm.is_female, (kid.DP / (mom.DP + dad.DP)) * (3/4)) # Because of hemi dad
                  .default( kid.DP / (mom.DP + dad.DP) ))

    def solve(p_de_novo, parent_tot_read_check, parent_alt_read_check):
        return (
            hl.case()
            .when(kid.GQ < min_gq, failure)
            .when((dp_ratio < min_dp_ratio)
                  | (kid_ad_ratio < min_child_ab), failure)
            .when(~parent_tot_read_check, failure)
            .when(~parent_alt_read_check, failure)
            .when(p_de_novo < min_p, failure)
            .when(~is_snp, hl.case()
                  .when((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (n_alt_alleles == 1),
                        hl.struct(p_de_novo=p_de_novo, confidence='HIGH'))
                  .when((p_de_novo > 0.5) & (kid_ad_ratio > 0.3) & (n_alt_alleles <= 5),
                        hl.struct(p_de_novo=p_de_novo, confidence='MEDIUM'))
                  .when(kid_ad_ratio > 0.2,
                        hl.struct(p_de_novo=p_de_novo, confidence='LOW'))
                  .or_missing())
            .default(hl.case()
                     .when(((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (dp_ratio > 0.2))
                           | ((p_de_novo > 0.99) & (kid_ad_ratio > 0.3) & (n_alt_alleles == 1))
                           | ((p_de_novo > 0.5) & (kid_ad_ratio > 0.3) & (n_alt_alleles < 10) & (kid.DP > 10)),
                           hl.struct(p_de_novo=p_de_novo, confidence='HIGH'))
                     .when((p_de_novo > 0.5) & ((kid_ad_ratio > 0.3) | (n_alt_alleles == 1)),
                           hl.struct(p_de_novo=p_de_novo, confidence='MEDIUM'))
                     .when(kid_ad_ratio > 0.2,
                           hl.struct(p_de_novo=p_de_novo, confidence='LOW'))
                     .or_missing()))
    
    # Call autosomal or pseudoautosomal de novo variants
    def call_auto(kid_pp, dad_pp, mom_pp):
        p_data_given_dn = dad_pp[0] * mom_pp[0] * kid_pp[1] * DE_NOVO_PRIOR
        p_het_in_parent = 1 - (1 - prior) ** 4
        p_data_given_missed_het = (dad_pp[1]*mom_pp[0] + dad_pp[0]*mom_pp[1]) * kid_pp[1] * p_het_in_parent
        # Note: a full treatment would include possibility kid genotype is in error:
        # "+ ( (dad_pp[0]*mom_pp[0] + dad_pp[0]*mom_pp[0]) * kid_pp[0] * (1 - p_het_in_parent) )",
        # which is approximately (dad_pp[0]*mom_pp[0] + dad_pp[0]*mom_pp[0]) * kid_pp[0]
        
        p_de_novo = p_data_given_dn / (p_data_given_dn + p_data_given_missed_het)

        parent_tot_read_check = (hl.sum(mom.AD) > 0) & (hl.sum(dad.AD) > 0)
        parent_alt_read_check = ( ((mom.AD[1] / hl.sum(mom.AD)) <= max_parent_ab) &
                                  ((dad.AD[1] / hl.sum(dad.AD)) <= max_parent_ab) )
            
        return hl.bind(solve, p_de_novo, parent_tot_read_check, parent_alt_read_check)

    # Call hemizygous de novo variants on the X
    def call_hemi_x(kid_pp, parent, parent_pp):
        p_data_given_dn = parent_pp[0] * kid_pp[2] * DE_NOVO_PRIOR
        p_het_in_parent = 1 - (1 - prior) ** 2
        p_data_given_missed_het = ((parent_pp[1] + parent_pp[2]) * kid_pp[2] * 
                                   p_het_in_parent) + (parent_pp[0] * kid_pp[0])
        # Note: if simplified to be more in line with the autosomal calls, this would be:
        # parent_pp[1] * kid_pp[2] * p_het_in_parent
        
        p_de_novo = p_data_given_dn / (p_data_given_dn + p_data_given_missed_het)

        parent_tot_read_check = (hl.sum(parent.AD) > 0)
        parent_alt_read_check = (parent.AD[1] / hl.sum(parent.AD)) <= max_parent_ab      

        return hl.bind(solve, p_de_novo, parent_tot_read_check, parent_alt_read_check)

    # Call hemizygous de novo variants on the Y and mitochondrial chromosome (ignoring mito heteroplasmy)
    def call_hemi_y_mt(kid_pp, parent, parent_pp):
        p_data_given_dn = parent_pp[0] * kid_pp[2] * DE_NOVO_PRIOR
        p_het_in_parent = 1 - (1 - prior) ** 1
        p_data_given_missed_het = (parent_pp[2] * kid_pp[2] * 
                                   p_het_in_parent) + (parent_pp[0] * kid_pp[0])
        # Note: if simplified to be more in line with the autosomal calls, this would be:
        # parent_pp[2] * kid_pp[2] * p_het_in_parent
        
        p_de_novo = p_data_given_dn / (p_data_given_dn + p_data_given_missed_het)

        parent_tot_read_check = (hl.sum(parent.AD) > 0)
        parent_alt_read_check = (parent.AD[1] / hl.sum(parent.AD)) <= max_parent_ab      

        return hl.bind(solve, p_de_novo, parent_tot_read_check, parent_alt_read_check)       
    
    # Main routine
    de_novo_call = (
        hl.case()
        .when(~has_candidate_gt_configuration, failure)
        .when(autosomal, hl.bind(call_auto, kid_pp, dad_pp, mom_pp))
        .when(hemi_x, hl.bind(call_hemi_x, kid_pp, mom, mom_pp))
        .when(hemi_y, hl.bind(call_hemi_y_mt, kid_pp, dad, dad_pp))
        .when(hemi_mt, hl.bind(call_hemi_y_mt, kid_pp, mom, mom_pp))
        .or_missing())

    tm = tm.annotate_entries(__call=de_novo_call)
    tm = tm.filter_entries(hl.is_defined(tm.__call))
    entries = tm.entries()
    return (entries.select('__site_freq',
                           'proband',
                           'father',
                           'mother',
                           'proband_entry',
                           'father_entry',
                           'mother_entry',
                           'is_female',
                           **entries.__call)
            .rename({'__site_freq': 'prior'}))
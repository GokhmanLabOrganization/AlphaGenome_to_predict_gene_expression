import re
import argparse
from Bio import SeqIO
import pandas as pd
import os
import polars as pl
import config
from alphagenome import colab_utils
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

FIELDS = [
        "ID",
        "Dbxref",
        "Name",
        "cell-line",
        "cell-type",
        "chromosome_name",
        "gbkey",
        "genome",
        "isolate",
        "mol_type",
        "sex",
        "tissue-type",
    ]

GFF_COLS_DICT = dict(zip(['column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 
                    'column_7', 'column_8', 'column_9'],
                     ['chromosome',
                      'source',
                            'feature',
                            'start',
                            'end',
                            'GeneName',
                            'strand',
                            'GeneID',
                            'details']))

def union_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """
    merging all intersecting exons.
    Args:
        intervals: List of (start, end) tuples representing exons.
    Returns:
        List of (start, end) tuples representing exons after merging of intersecting intervals.
    """
    if not intervals:
        return []

    # sort by start, then end
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:                 # overlap (use s <= me+1 if you want to merge "touching" for ints)
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))

    return merged

def get_sequence_records(genome_file):
    """
    Load FASTA records for a genome file from the shared genomes directory.
    Args:
        genome_file: Genome FASTA filename relative to the genomes directory.
    Returns:
        A list of Biopython ``SeqRecord`` objects.
    """
    genomes_directory = '/home/labs/davidgo/Collaboration/Genomes/Genome_fastas/'
    genome_full_path = genomes_directory + genome_file
    if os.path.exists(genome_full_path):
        records = list(SeqIO.parse(genome_full_path, "fasta"))
        print(f'Imported genome {genome_full_path}')
    else:
        print(f'No Genome file {genome_full_path}')
        return
    return records

def seq_retrieve_interval(chromosome, start, end, records):
    """Extract a chromosome interval and pad it to the AlphaGenome 1 Mb input size.

    Args:
        chromosome: Chromosome/contig identifier matching ``record.id``.
        start: 1-based inclusive genomic start coordinate.
        end: 1-based inclusive genomic end coordinate.
        records: Iterable of sequence records for the reference genome.

    Returns:
        Uppercase DNA sequence centered and padded with ``N`` to
        ``dna_client.SEQUENCE_LENGTH_1MB``.
    """

    curr_chr = [record for record in records if record.id == chromosome] # this is how to exctract the record that matches our chromosome
    sequence = str(curr_chr[0].seq[start - 1:end].upper()) # note - the coords here are similar to bed format: start is -1 (starting with 0), and end is not included
    mb_seq = sequence.center(dna_client.SEQUENCE_LENGTH_1MB, 'N')
    return mb_seq


def get_gene_1mb_region(gene_id, gene_gff_df):
    """Compute a 1 Mb window and metadata for a gene.

    Args:
        gene_id: Gene identifier to locate in ``gene_gff_df``.
        gene_gff_df: Polars DataFrame with at least ``GeneID``, ``Name``,
            ``chromosome``, ``start``, ``end``, and ``strand``.

    Returns:
        Tuple ``(gene_symbol, gene_chr, gene_start_1mb, gene_end_1mb, gene_strand)``.
        If the gene is missing, returns ``(None, None, None, None, None)``.
    """
    gene_gff = gene_gff_df.filter(((pl.col('GeneID') == gene_id)))
    if gene_gff.height == 0:
        return None, None, None, None, None
    gene_symbol = gene_gff['Name'].item()
    gene_strand = gene_gff.select(pl.col('strand')).item()
    gene_chr = gene_gff.select(pl.col('chromosome')).item()
    gene_start = gene_gff.select(pl.col('start')).item()
    gene_end = gene_gff.select(pl.col('end')).item()
    gene_start_1mb = gene_gff.select(pl.col('start')).item() - (2**19)
    gene_end_1mb = gene_gff.select(pl.col('end')).item() + (2**19) - (gene_end - gene_start +1)
    
    return gene_symbol , gene_chr, gene_start_1mb, gene_end_1mb, gene_strand

def prepare_gff(gff_df, fields):
    """Expand semicolon-delimited GFF attributes into structured columns.

    Args:
        gff_df: Raw GFF-like DataFrame that includes a ``details`` column.
        fields: Ordered attribute names to map onto parsed ``details`` values.

    Returns:
        DataFrame with parsed attributes unnested into dedicated columns.
    """
    gff_with_split = gff_df.with_columns(pl.col('details').str.split(';'))
    gff_with_split = gff_with_split.with_columns(pl.col('details').map_elements(lambda x: [field.split('=')[1] for field in x], return_dtype = pl.List(pl.String)))
    # return gff_with_split
    # gff_with_split_clean = gff_with_split.with_columns(pl.col('details').map_elements(lambda x: [field.split('=')[1] for field in x], return_dtype = pl.List(pl.String)))
    gff_struct = gff_with_split.with_columns(pl.col('details').list.to_struct(fields= fields))
    gff_final = gff_struct.unnest('details')
    return gff_final

# def make_gff_usable(func):
#     """Wrapper utility that executes a GFF-preparation callable.

#     Args:
#         func: Callable that returns prepared GFF DataFrames.

#     Returns:
#         The value returned by ``func``.
#     """
#     return func()
    
def make_chimp_gff_usable():
    """Load and split chimp GFF table into gene and exon records.

    Returns:
        Tuple ``(genes_gff, exons_gff)`` as Polars DataFrames.
    """
    chimp_gff = pl.read_csv(config.CHIMP_GFF_WEXAC, separator= '\t', comment_prefix='#', has_header= False)
    chimp_gff_renamed = chimp_gff.rename(GFF_COLS_DICT)
    genes_gff ,exons_gff= chimp_gff_renamed.filter(pl.col('feature') == 'gene'), chimp_gff_renamed.filter(pl.col('feature') == 'exon')
    return genes_gff ,exons_gff

def AG_gene_prediction(AG_model, gtf , seq, gene_symbol, show= False):
    """Run AlphaGenome RNA-seq prediction for a sequence window.

    Args:
        AG_model: Initialized AlphaGenome DNA client.
        gtf: GTF annotation table used for interval lookup/plotting.
        seq: Input DNA sequence (expected 1 Mb length).
        gene_symbol: Gene symbol used for interval retrieval when ``show=True``.
        show: If ``True``, predicts with an explicit gene interval and renders a
            transcript + RNA track plot.

    Returns:
        AlphaGenome prediction object containing RNA-seq tracks.
    """

    if not show:
        
        rna_seq_pred = AG_model.predict_sequence(
            sequence=seq, 
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=['CL:0000138'],  # CHONDROCYTES.
        )
    
    
    if show:
        gene_interval = gene_annotation.get_gene_interval(gtf, gene_symbol).resize(dna_client.SEQUENCE_LENGTH_1MB)
        rna_seq_pred = AG_model.predict_sequence(
            sequence=seq, 
            interval=gene_interval,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=['CL:0000138'],  # CHONDROCYTES.
        )
        gtf_transcripts = gene_annotation.filter_protein_coding(gtf)
        gtf_transcripts = gene_annotation.filter_to_mane_select_transcript(gtf_transcripts)
        transcript_extractor = transcript_utils.TranscriptExtractor(gtf_transcripts)
        transcripts = transcript_extractor.extract(gene_interval)
        plot = plot_components.plot(
            [plot_components.TranscriptAnnotation(
                    transcripts, fig_height=0.1
                ),
                plot_components.Tracks(
                    tdata=rna_seq_pred.rna_seq,
                    ylabel_template='RNA_SEQ: {biosample_name} ({strand})\n{name}',
                )
            ],
            interval = gene_interval
        )
        plot.show()
    
    return rna_seq_pred

def human_chimp_lfc(human_gene_id, chimp_gene_id, human_records, chimp_records, human_gene_gff, human_exons_gff, chimp_gene_gff, chimp_exons_gff, ag_model, gtf):
    """Compute human-vs-chimp LFC for one orthologous gene pair.

    Args:
        human_gene_id: Human gene ID.
        chimp_gene_id: Chimp gene ID.
        human_records: Human genome FASTA records.
        chimp_records: Chimp genome FASTA records.
        human_gene_gff: Human gene annotation DataFrame.
        human_exons_gff: Human exon annotation DataFrame.
        chimp_gene_gff: Chimp gene annotation DataFrame.
        chimp_exons_gff: Chimp exon annotation DataFrame.
        ag_model: Initialized AlphaGenome client.
        gtf: GTF annotation table.

    Returns:
        Tuple ``(lfc, human_expr, chimp_expr, human_symbol, chimp_symbol)``.
        ``lfc`` is ``None`` when either species expression cannot be computed.
    """
    human_gene_expression, human_gene_symbol = calc_gene_expression(human_gene_id, human_records,human_gene_gff,human_exons_gff, ag_model, gtf)
    chimp_gene_expression, chimp_gene_symbol = calc_gene_expression(chimp_gene_id, chimp_records, chimp_gene_gff, chimp_exons_gff, ag_model, gtf)
    if chimp_gene_expression == None or human_gene_expression == None:
        return None, human_gene_expression, chimp_gene_expression, human_gene_symbol, chimp_gene_symbol
    lfc = np.log2(human_gene_expression/chimp_gene_expression)
    return lfc, human_gene_expression, chimp_gene_expression , human_gene_symbol, chimp_gene_symbol

def calc_gene_expression(gene_id, records ,gene_gff, exon_gff, ag_model, gtf, prediction_func = AG_gene_prediction):
    """Predict and summarize expression for a single gene.

    Args:
        gene_id: Gene ID to evaluate.
        records: Genome FASTA sequence records for the species.
        gene_gff: Gene annotation DataFrame.
        exon_gff: Exon annotation DataFrame.
        ag_model: Initialized AlphaGenome client.
        gtf: GTF annotation table.

    Returns:
        Tuple ``(gene_expression, gene_symbol)`` where expression is strand-aware
        mean RNA-seq signal across the union of exon intervals.
        Returns ``(None, None)`` when the gene cannot be found.
    """
    gene_symbol, gene_chr, gene_start_1mb, gene_end_1mb, gene_strand = get_gene_1mb_region(gene_id, gene_gff)
    if gene_symbol is None:
        print(f'Gene {gene_id} not found in gff')
        return None, None
    gene_1mb_seq = seq_retrieve_interval(gene_chr, gene_start_1mb, gene_end_1mb, records)
    full_pred = prediction_func(ag_model, gtf, gene_1mb_seq, gene_symbol, show = False)
    gene_expression = coding_region_mean_expression(exon_gff, gene_id, gene_strand, gene_start_1mb, full_pred)
    return gene_expression , gene_symbol

def make_human_gff_usable():
    """Load and normalize human gene/exon annotation tables.

    Returns:
        Tuple ``(human_gene_gff, human_exon_gff)`` as Polars DataFrames.
    """
    human_gene_gff = pl.read_csv('/home/labs/davidgo/Collaboration/GenomeAnnotation/Human/NCBI/hg38/gff_parsed/NCBI_genes_hg38.txt', separator= '\t', comment_prefix='#', has_header= True).rename({'GeneID': 'gff_ID'}).with_columns(pl.col('gff_ID').cast(pl.Utf8))
    human_exon_gff = pl.read_csv('/home/labs/davidgo/Collaboration/GenomeAnnotation/Human/NCBI/hg38/gff_parsed/NCBI_exons_hg38.txt', separator= '\t', comment_prefix='#', has_header= True).drop('frame').rename({'transcript_id': 'gff_ID'})
    human_exon_gff = human_exon_gff.with_columns(pl.lit('exon').alias('feature'))
    # human_gff = pl.concat([human_gene_gff, human_exon_gff], how = 'vertical')
    return human_gene_gff, human_exon_gff

def coding_region_mean_expression(exon_gff, gene_ID, gene_strand, gene_start_1mb, full_pred):
    """Aggregate predicted RNA-seq signal over merged exon intervals.

    Args:
        exon_gff: Exon annotation DataFrame.
        gene_ID: Gene identifier used to match exon rows.
        gene_strand: ``'+'`` or ``'-'``; determines which RNA channel to use.
        gene_start_1mb: Start coordinate of the 1 Mb model input window.
        full_pred: AlphaGenome prediction object with ``rna_seq.values``.

    Returns:
        Mean coding-region expression value for the matching strand,
        or ``None`` when no exonic bases are found.
    """
    gene_exon_gff = exon_gff.filter(pl.col('Name').str.contains(gene_ID) | pl.col('Dbxref').str.contains(gene_ID))
    exon_intervals = list(zip(gene_exon_gff['start'], gene_exon_gff['end']))
    exon_intervals_union = union_intervals(exon_intervals)
    coding_region_pred = [full_pred.rna_seq.values[i] for s, e in exon_intervals_union for i in range(max(0, s-gene_start_1mb), min(1048576, e-gene_start_1mb))]
    if len(coding_region_pred) == 0:
        return None
    mean_gene_expression_strand_p, mean_gene_expression_strand_n =  np.mean(coding_region_pred, axis=0)
    if gene_strand == '+':
        return mean_gene_expression_strand_p
    elif gene_strand == '-':
        return mean_gene_expression_strand_n


def go_over_all_genes(start_i = None, end_i= None):
    """Process a chunk of orthologous genes and compute per-gene metrics.

    Args:
        start_i: Start index (inclusive) in the ortholog gene list.
        end_i: End index (exclusive) in the ortholog gene list.

    Returns:
        Polars DataFrame with columns for gene symbols/IDs, LFC, and species
        expression values for each processed ortholog pair.
    """
    chimp_records = get_sequence_records(config.CHIMP_RECORDS)
    human_records = get_sequence_records(config.HUMAN_RECORDS)
    chimp_gene_gff, chimp_exons_gff = pl.read_csv(config.CHIMP_GENES_GFF_PATH_WEXAC, separator = '\t', ignore_errors=True), pl.read_csv(config.CHIMP_EXONS_GFF_PATH_WEXAC, separator = '\t', ignore_errors=True)
    human_gene_gff, human_exons_gff = pl.read_csv(config.HUMAN_GENES_GFF_PATH_WEXAC, separator = '\t', ignore_errors=True), pl.read_csv(config.HUMAN_EXONS_GFF_PATH_WEXAC, separator = '\t', ignore_errors=True) 
    AG_model = dna_client.create(config.AG_API_KEY)
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    human_chimp_gene_ids = list(zip(human_gene_gff['GeneID'], human_gene_gff['chimp_ortholog_GeneID']))
    human_chimp_gene_ids = human_chimp_gene_ids[start_i:end_i]
    rows = []
    
    for human_gene_id, chimp_gene_id in human_chimp_gene_ids:
        gene_lfc, human_gene_expression, chimp_gene_expression, human_gene_symbol, chimp_gene_symbol = human_chimp_lfc(human_gene_id, chimp_gene_id, human_records, chimp_records, human_gene_gff, human_exons_gff, chimp_gene_gff, chimp_exons_gff, AG_model, gtf)
        
        rows.append({"GeneSymbol": human_gene_symbol, "HumanGeneID": human_gene_id, "ChimpGeneID": chimp_gene_id, "LFC": gene_lfc, "HumanGeneExpression": human_gene_expression, "ChimpGeneExpression": chimp_gene_expression})

    lfc_df = pl.DataFrame(rows)
    return lfc_df
    
    
def main():
    """CLI entrypoint for chunked human/chimp expression prediction.

    Command-line arguments:
        --job_i: Zero-based chunk index.
        --chunk_size: Number of genes per chunk (default: 500).
        --output_dir: Directory to write results (default: results/AG_predictions_<timestamp>).
    """
    from datetime import datetime

    parser = argparse.ArgumentParser()

    parser.add_argument("--job_i", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    start_i = args.job_i * args.chunk_size
    end_i = start_i + args.chunk_size
    print(f'Processing genes from index {start_i} to {end_i}')
    lfc_df = go_over_all_genes(
        start_i=start_i,
        end_i=end_i
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, "results", f"AG_predictions_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f'Finished processing genes from index {start_i} to {end_i}, saving results to {output_dir}')
    lfc_df.write_csv(f'{output_dir}/lfc_df_{start_i}_{end_i}.tsv', separator= '\t')
    
if __name__ == "__main__":
    print("Entered main()")
    main()
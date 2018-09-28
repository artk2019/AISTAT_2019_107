# reads_fn="/home/arecanat/Desktop/test_genome_exp/ecoli_data/reads.fasta"
# ovlp_fn="/home/arecanat/Desktop/test_genome_exp/results/ovlp.paf"
# aln_fn="/home/arecanat/Desktop/test_genome_exp/results/aln.sam"
# res_dir="/home/arecanat/Desktop/test_genome_exp/results/"
# python_script="/home/arecanat/Dropbox/RobustSeriationEmbedding/mdso/mdso/examples/ecoli/run_genome_exp_clean.py"

reads_fn="/Users/antlaplante/THESE/RobustSeriationEmbedding/test_genome_exp/ecoli_data/reads.fasta"
ovlp_fn="/Users/antlaplante/THESE/RobustSeriationEmbedding/test_genome_exp/results/ovlp.paf"
aln_fn="/Users/antlaplante/THESE/RobustSeriationEmbedding/test_genome_exp/results/aln.sam"
res_dir="/Users/antlaplante/THESE/RobustSeriationEmbedding/test_genome_exp/results/"
python_script="/Users/antlaplante/Dropbox/RobustSeriationEmbedding/mdso/examples/ecoli/run_genome_exp_clean.py"

for k_nbrs in 15 30 60
do
  for dim in 5 15 30
  do
    for ovlp_thr_qtle in 10 30 40 50 60 70 80
    do
      python ${python_script} ${reads_fn} ${ovlp_fn} ${aln_fn} -r ${res_dir} -k ${k_nbrs} -d ${dim} -t ${ovlp_thr_qtle}
    done
  done
done

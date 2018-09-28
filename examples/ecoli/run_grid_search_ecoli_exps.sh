#!/bin/bash
export SGE_ROOT=/cm/shared/apps/sge/2011.11p1;
COUNTERJOBS=`/cm/shared/apps/sge/2011.11p1/bin/linux-x64/qstat -u arecanat | wc -l`
memory=3500M
vmemory=3800M
nb_node=22

reads_fn="/sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp/ecoli_data/reads.fasta"
ovlp_fn="/sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp/results/ovlp.paf"
aln_fn="/sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp/results/aln.sam"
res_dir="/sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp/results/"
python_script="/sequoia/data1/arecanat/RobustSeriationEmbedding/mdso/examples/ecoli/run_genome_exp_clean.py"

cd /sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp

exp_dir="/sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp/results/"

for k_nbrs in 5 15 30 60; do
  for dim in 3 20; do
    for ovlp_thr_qtle in 20 30 40 45 50 55 60 65 70 80; do
        COUNTERJOBS=`qstat -u arecanat | wc -l`
        echo "  job count : ${COUNTERJOBS}"
        while [ $COUNTERJOBS -ge $nb_node ]; do
            sleep 10
            COUNTERJOBS=`qstat -u arecanat | wc -l`
        done

        NAME="seriation"_"$k_nbrs"_"$dim"_"$ovlp_thr_qtle"
        echo "#$ -o /sequoia/data1/arecanat/RobustSeriationEmbedding/logs/$NAME.out
              #$ -e /sequoia/data1/arecanat/RobustSeriationEmbedding/logs/$NAME.err
              #$ -l mem_req=${memory},h_vmem=${vmemory}
              #$ -N $NAME
              #$ -q all.q,goodboy.q
              #$ -pe serial 2

              echo 00
              export PATH=/sequoia/data1/arecanat/anaconda3/bin:/sequoia/data1/arecanat/software/jre1.8.0_121/bin:/sequoia/data1/arecanat/anaconda2/bin:/cm/shared/apps\
              /sge/2011.11p1/bin/linux-x64:/cm/shared/apps/gcc/4.8.1/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/sbin:/usr/\
              sbin:/opt/dell/srvadmin/bin:/home/arecanat/bin:$PATH
              export MKL_NUM_THREADS=2
              export NUMEXPR_NUM_THREADS=2
              export OMP_NUM_THREADS=2
              cd /sequoia/data1/arecanat/RobustSeriationEmbedding/test_genome_exp/results/
              python ${python_script} ${reads_fn} ${ovlp_fn} ${aln_fn} -r ${res_dir} -k ${k_nbrs} -d ${dim} -t ${ovlp_thr_qtle}
              echo 11
              " | sed "s/^ *//g" > /sequoia/data1/arecanat/RobustSeriationEmbedding/logs/$NAME.pbs

        qsub /sequoia/data1/arecanat/RobustSeriationEmbedding/logs/$NAME.pbs
      done
    done
  done

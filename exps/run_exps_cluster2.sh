#!/bin/bash
export SGE_ROOT=/cm/shared/apps/sge/2011.11p1;
COUNTERJOBS=`/cm/shared/apps/sge/2011.11p1/bin/linux-x64/qstat -u arecanat | wc -l`
memory=3500M
vmemory=3800M
nb_node=24

pythonfile="/sequoia/data1/arecanat/RobustSeriationEmbedding/mdso/mdso/exps/parse_args_run_exps.py"
cd /sequoia/data1/arecanat/RobustSeriationEmbedding/

exp_dir="/sequoia/data1/arecanat/RobustSeriationEmbedding/temp_exp_res/"
doScale=1
kNeighbors=10
# for typeLapEmbed in random_walk unnormalized; do
#   for typeLapFiedler in unnormalized random_walk; do
    for doScale in 2; do
      for kNeighbors in 5 7 10 15 20 30; do
      for dim in 10; do
        for typeMatrix in LinearBanded LinearStrongDecrease CircularBanded CircularStrongDecrease; do
          for amplNoise in 0. 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
            COUNTERJOBS=`qstat -u arecanat | wc -l`
            echo "  job count : ${COUNTERJOBS}"
            while [ $COUNTERJOBS -ge $nb_node ]; do
                sleep 10
                COUNTERJOBS=`qstat -u arecanat | wc -l`
            done

            NAME="seriation_$doScale"_"$kNeighbors"_"$dim"_"$typeMatrix"_"$amplNoise"
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
                  cd /sequoia/data1/arecanat/RobustSeriationEmbedding/
                  python ${pythonfile} -r ${exp_dir} -s ${doScale} -m ${typeMatrix} -a ${amplNoise} -k ${kNeighbors} -d ${dim}
                  echo 11
                  " | sed "s/^ *//g" > /sequoia/data1/arecanat/RobustSeriationEmbedding/logs/$NAME.pbs

            qsub /sequoia/data1/arecanat/RobustSeriationEmbedding/logs/$NAME.pbs
          done
        done
      done
    done
    done
#   done
# done

export FLIP=pos
export NREV=3
export NSAM=2
qsub -V run.qsub

for FLIP in pos neg
do
  for NREV in 1 2 3 5 7 10 15 25 40 60 80 100 150 250 400 600 900 1000
  do 
    export FLIP
    export NREV
    export NSAM=100
    echo "submitting for ${FLIP} with ${NREV} reviews for ${NSAM} samples"
    qsub -V run.qsub
  done
done
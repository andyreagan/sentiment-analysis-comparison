cd /users/a/r/areagan/work/2015/03-sentiment-analysis-comparison
/gpfs1/arch/x86_64/python2.7.5/bin/python qsub.py
grep -l "job completion success" explode* | xargs rm

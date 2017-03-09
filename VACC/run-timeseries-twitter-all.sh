# run all of the twitter timeseries generation, full timeseries
#
# example call:
# /usr/bin/time -v python analyzeAll.py twitter-timeseries 2015-03-01 ${RES} ${DICT} ${STOPVAL}

for DICT in 0 1 2
do
    export DICT=$DICT
    for RES in 1 2 3 4
    do
        export RES=$RES
        for STOPVAL in 0.5 1.0 1.5 2.0
        do
            export STOPVAL=$STOPVAL
	    qsub -qshortq -V run-timeseries-twitter.qsub
        done
    done
done

for DICT in 3 4 5
do
    export DICT=$DICT
    for RES in 0 1 2 3 4
    do
        export RES=$RES
	export STOPVAL=0.5
	qsub -qshortq -V run-timeseries-twitter.qsub
    done
done

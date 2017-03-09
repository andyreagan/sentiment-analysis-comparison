################################################################################
# movie reviews
#
# pre-processing:
# unzip the files in data/moviereviews
#
# cleaning:
# stored results are all in a folder in the output/ folder, named
# by the test

# export NSAM=100

# for FLIP in pos neg
# # for FLIP in neg
# do
#   for NREV in 1 2 3 5 7 10 15 25 40 60 80 100 150 250 400 600 900 1000
#   # for NREV in 100 150 250 400 600 900 1000
#   do 
#     export FLIP
#     export NREV
#     echo "running for ${FLIP} with ${NREV} reviews for ${NSAM} samples"
#     echo "python analyzeAll.py reviewTest ${FLIP} ${NREV} ${NSAM}"
#     python3 analyzeAll.py reviewTest ${FLIP} ${NREV} ${NSAM}
#   done
# done

# python analyzeAll.py movie-review-test

# python analyzeAll.py plot-movie-reviews

python analyzeAll.py review-wordshifts

# run these two locally
cd figures/moviereviews-shifts
. convert.sh
. combine.sh
cd ../../

################################################################################
# new york times
#
# pre-processing:
# starting with jake's section-parsed data
#
# cleaning:
# this process will save pickles of sections words as dicts in the .dict file,
# and a .csv for every dictionary
# clear all of those, and they will automatically be remade
# note: clear ALL of them, not just parts. checks for the all.dict file.

python analyzeAll.py nyt-all

cd figures/nyt-shifts
. combine.sh
cd ../..

# this was just a test
# python3 analyzeAll.py shift-NYT-zipf

################################################################################
# twitter
#
# pre-processing:
# on the VACC, run this thing
# python3 analyzeAll.py twitter-timeseries
# python3 analyzeAll.py twitter-addyears
#
# cleanup:
# make years files on the VACC (above), zip up everything on the VACC except years
#
# reprocessing:
# run on the vacc to make just the daily resolution files
# zip up everything there into the 6 zip files using a shell one-liner:
#     for dict in LabMT ANEW WK MPQA LIWC Liu; do for year in 2008 2009 20{10..15}; do zip ${dict}.zip word-vectors/${year}*/*-${dict}.csv; done; done;
# copy those down into the data/twitter folder, and unzip
# use the updated twitter-addyears which calls the local function to add them up

# python analyzeAll.py twitter-plot-timeseries

# this needs the timeseries for the full resolution
# python analyzeAll.py twitter-correlations

python analyzeAll.py twitter-shiftyears

cd figures/twitter-shifts
. combine.sh
cd ../..

################################################################################
# google books

# python analyzeAll.py plot-google-books

python analyzeAll.py plot-google-books-shifts
cd figures/googlebooks-shifts
. combine.sh
cd ../../

# ################################################################################
# # coverage

# python3 analyzeAll.py coverage

# ################################################################################
# # bayes

# python3 analyzeAll.py moviereviews-bayes

# python3 analyzeAll.py bayes-nyt

# python3 analyzeAll.py bayes-shift

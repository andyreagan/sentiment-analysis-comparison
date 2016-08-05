# analyzeAll.py
#
# make the whole paper!
#
# USAGE
#
# python analyzeAll.py coverage
# python analyzeAll.py reviewTest [pos,neg] [numReviews] [numSamples]
# python analyzeAll.py plotClass


# load the very latest version
import sys
sys.path.append("/Users/andyreagan/tools/python/labMTsimple/")
# for the VACC, doesn't hurt to have both
sys.path.append("/users/a/r/areagan/work/2014/03-labMTsimple/")
from labMTsimple.speedy import *
from labMTsimple.storyLab import *

import re
import codecs
from os import listdir,mkdir
from os.path import isfile,isdir
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
rc("xtick", labelsize=8)
rc("ytick", labelsize=8)
rc("font",**{"family":"serif","serif":["cmr10"]})
# rcParams["mathtext.fontset"] = "stix"
# rcParams["font.family"] = "STIXGeneral"
rc("text", usetex=True)
figwidth_onecol = 8.5
figwidth_twocol = figwidth_onecol/2

import numpy as np
from json import loads
import csv
from datetime import datetime,timedelta
import pickle

from subprocess import call

from scipy.stats import pearsonr

error_logging = True
sys.path.append("/Users/andyreagan/tools/python/kitchentable")
from dogtoys import *

def loadMovieReviews():
    posfiles = ["data/moviereviews/txt_sentoken/pos/"+x for x in listdir("data/moviereviews/txt_sentoken/pos") if ".txt" in x]
    negfiles = ["data/moviereviews/txt_sentoken/neg/"+x for x in listdir("data/moviereviews/txt_sentoken/neg") if ".txt" in x]
    poswordcounts = dict()
    allwordcounts = dict()
    for file in posfiles:
        f = open(file,"r")
        postext = f.read() + " "
        f.close()
        dictify_general(postext,poswordcounts)
        dictify_general(postext,allwordcounts)
    negwordcounts = dict()
    for file in negfiles:
        f = open(file,"r")
        negtext = f.read() + " "
        f.close()
        dictify_general(negtext,negwordcounts)
        dictify_general(negtext,allwordcounts)

    print("there are {0} unique words in this corpus".format(len(allwordcounts)))

    # rip those dictionaries into lists for sorting
    allwordsList = [word for word in allwordcounts]
    allcountsList = [allwordcounts[allwordsList[i]] for i in range(len(allwordsList))]

    # sort them
    indexer = sorted(range(len(allcountsList)), key=lambda k: allcountsList[k], reverse=True)
    allcountsListSorted = np.array([float(allcountsList[i]) for i in indexer])
    allwordsListSorted = [allwordsList[i] for i in indexer]

    return allcountsListSorted,allwordsListSorted

def loadMovieReviewsBoth():
    flip = "pos"
    pos_files = ["data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
             for x in listdir("data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
    pos_wordcounts = dict()
    for file in pos_files:
        # this loads the files
        f = open(file+".txt","r")
        rawtext = f.read()
        f.close()
        # add to the full dict
        dictify_general(rawtext,pos_wordcounts)
    flip = "neg"
    neg_files = ["data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
             for x in listdir("data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
    neg_wordcounts = dict()
    for file in neg_files:
        # this loads the files
        f = open(file+".txt","r")
        rawtext = f.read()
        f.close()
        # add to the full dict
        dictify_general(rawtext,neg_wordcounts)

    return (pos_wordcounts,neg_wordcounts)

word_dict_location = "data/twitter/word-dicts/"
word_vector_location = "data/twitter/word-vectors/"
# word_dict_location = "/users/a/r/areagan/scratch/realtime-parsing/word-dicts/"
# word_vector_location = "/users/a/r/areagan/scratch/realtime-parsing/word-vectors/"


def loadTweetsDynamic(start,end):
    """Load all of the word dicts up to, but not including, the one at end.

    Takes two datetime objects, and loads lots of dicts (potentially), adding them up."""

    sum_dict = dict()
    resolution = timedelta(minutes=15)
    curr_date = start
    while curr_date < end:
        fname = curr_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M.dict".format(word_dict_location))
        print("loading {0}".format(fname))
        found_file = True
        try:
            tmp_dict = pickle.load( open( fname , "rb" ) )
        except IOError:
            found_file = False
        if found_file:
            for word in tmp_dict:
                if word in sum_dict:
                    sum_dict[word] += tmp_dict[word]
                else:
                    sum_dict[word] = tmp_dict[word]
        else:
            print("did not find {0}".format(fname))

        curr_date += resolution
    return sum_dict

def make_twitter_wordvecs_day(date):
    """Load the dicts for that day, and turn it into a word vec for each dictionary.
    Saves the csv's at daily resolution.

    """

    stopVal = 0.0
    
    allDicts = (ANEW(stopVal=stopVal),
                LIWC(stopVal=stopVal),
                MPQA(stopVal=stopVal),
                Liu(stopVal=stopVal),
                WK(stopVal=stopVal))

    my_full_vecs = [np.zeros(len(my_senti_dict.wordlist)) for my_senti_dict in allDicts]

    resolution = timedelta(minutes=15)
    curr_date = date
    while curr_date < date+timedelta(days=1):
        fname = curr_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M.dict".format(word_dict_location))
        print("attempting to load {0}".format(fname))
        found_file = True
        try:
            tmp_dict = pickle.load( open( fname , "rb" ) )
            print("found")
        except IOError:
            print("did not find {0}".format(fname))
            found_file = False
        if found_file:
            for i,my_senti_dict in enumerate(allDicts):
                my_wordvec = my_senti_dict.wordVecify(tmp_dict)
                my_full_vecs[i] += my_wordvec

        curr_date += resolution
        
    for i,my_senti_dict in enumerate(allDicts):
        f = open(date.strftime("{0}%Y-%m-%d/%Y-%m-%d-all-{1}.csv".format(word_vector_location,my_senti_dict.corpus)),"w")
        f.write("\n".join(list(map(lambda x: "{0:.0f}".format(x),my_full_vecs[i]))))
        f.close()
    

def make_twitter_wordvecs_full_res(start,end):
    """Load all of the word dicts and convert them into word-vecs for every dictionary.
    Saves the csv's at 15min resolution.

    Takes two datetime objects, and loads lots of dicts (potentially)."""

    stopVal = 0.0
    # allDicts = (LabMT(stopVal=stopVal),
    #             ANEW(stopVal=stopVal),
    #             LIWC(stopVal=stopVal),
    #             MPQA(stopVal=stopVal),
    #             Liu(stopVal=stopVal),
    #             WK(stopVal=stopVal))

    allDicts = (ANEW(stopVal=stopVal),
                LIWC(stopVal=stopVal),
                MPQA(stopVal=stopVal),
                Liu(stopVal=stopVal),
                WK(stopVal=stopVal))

    resolution = timedelta(minutes=15)
    curr_date = start
    while curr_date < end:
        fname = curr_date.strftime("%Y-%m-%d/%Y-%m-%d-%H-%M.dict".format(word_dict_location))
        # on the VACC, folders exist
        # if not isdir(curr_date.strftime("{0}%Y-%m-%d/".format(word_vector_location))):
        #     mkdir(curr_date.strftime("{0}%Y-%m-%d/".format(word_vector_location)))
        print("attempting to load {0}".format(fname))
        found_file = True
        try:
            tmp_dict = pickle.load( open( fname , "rb" ) )
        except IOError:
            print("did not find {0}".format(fname))
            found_file = False
        if found_file:
            for i,my_senti_dict in enumerate(allDicts):
                my_wordvec = my_senti_dict.wordVecify(tmp_dict)
                f = open(curr_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M-{1}.csv".format(word_vector_location,my_senti_dict.corpus)),"w")
                f.write("\n".join(list(map(lambda x: "{0:.0f}".format(x),my_wordvec))))
                f.close()

        curr_date += resolution

def twitter_timeseries_all(start,end):
    """Make a full twitter timeseries with different resolutions, using word vecs."""

    # resolutions = [timedelta(minutes=15),timedelta(hours=1),timedelta(hours=3),timedelta(hours=12),timedelta(days=1),]
    resolutions = [timedelta(days=1),]

    stopVal = 0.0
    allDicts = (LabMT(stopVal=stopVal),
                LabMT(stopVal=stopVal),
                LabMT(stopVal=stopVal),
                ANEW(stopVal=stopVal),
                ANEW(stopVal=stopVal),
                ANEW(stopVal=stopVal),
                WK(stopVal=stopVal),
                WK(stopVal=stopVal),
                WK(stopVal=stopVal),
                LIWC(stopVal=stopVal),
                MPQA(stopVal=stopVal),
                Liu(stopVal=stopVal),)

    all_stopVals = [0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5,1.0,1.0,1.0]

    # a timeseries for each resolution, for each dictionary
    all_timeseries = [[[] for j in range(len(allDicts))] for i in range(len(resolutions))]
    all_times = [[] for i in range(len(resolutions))]

    # loop over all resolutions
    for i,resolution in enumerate(resolutions):
        print("resolution {0}".format(i))
        # loop over all dictionaries
        for j,my_senti_dict in enumerate(allDicts):
            print("dictionary {0}".format(j))
            times,timeseries = twitter_timeseries_day_local(start,end,resolution,my_senti_dict,all_stopVals[j])
            # times,timeseries = twitter_timeseries(start,end,resolution,my_senti_dict,all_stopVals[j])
            all_timeseries[i][j].append(timeseries)
            if j==0:
                all_times[i].append(times)

    return (all_times,all_timeseries)

def twitter_addyears(my_senti_dict):
    '''Make single wordvecs for each year, using word vectors on the VACC.'''
    times = []
    timeseries = []

    base_resolution = timedelta(minutes=15)

    for year in range(2008,2016):
        my_word_vec = np.zeros(len(my_senti_dict.wordlist))
        
        start = datetime(year,1,1)
        end = datetime(year+1,1,1)        
        curr_date = start
        
        while curr_date < end:
            if my_senti_dict.corpus == "LabMT":
                 fname = curr_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M.csv".format(word_vector_location))
            elif my_senti_dict.corpus == "WK":
                 fname = curr_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M-Warriner.csv".format(word_vector_location))
            else:
                 fname = curr_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M-{1}.csv".format(word_vector_location,my_senti_dict.corpus))
            print(fname)
            if isfile(fname):
                f = open(fname,"r")
                tmp_word_vec = np.array(map(float,f.read().split("\n")))
                f.close()
                my_word_vec += tmp_word_vec
                # if sum(tmp_word_vec) == 0:
                #     print("empty file at {}".format(fname))
            # else:
            #     print("missing {}".format(fname))

            curr_date += base_resolution

        print("done with year {}, saving".format(year))

        f = open("/users/a/r/areagan/work/2015/03-sentiment-analysis-comparison/data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"w")
        f.write("\n".join(list(map(lambda x: "{0:.0f}".format(x),my_word_vec))))
        f.close()

def twitter_addyears_local(my_senti_dict):
    '''Make single wordvecs for each year, using word vectors on the VACC.'''
    times = []
    timeseries = []

    resolution = timedelta(days=1)
    error_logging = False

    for year in range(2008,2016):
        my_word_vec = np.zeros(len(my_senti_dict.wordlist))
        # print(len(my_word_vec))
        # print(my_senti_dict.wordlist)
        start = datetime(year,1,1)
        end = datetime(year+1,1,1)

        times,timeseries,my_word_vec = twitter_timeseries_day_local(start,end,resolution,my_senti_dict,stopVal,return_total=True)

        f = open("data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"w")
        f.write("\n".join(list(map(lambda x: "{0:.0f}".format(x),my_word_vec))))
        f.close()

def twitter_shiftyears(my_senti_dict,stopVal=0.0,j=0):
    '''Using the twitter yearly wordvecs locally in data/twitter/, make
    a shift of each year against the background.'''
    
    my_full_vec = np.zeros(len(my_senti_dict.wordlist))

    for year in range(2008,2016):
        print("opening year {} to add it up".format(year))
        f = open("data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"r")
        my_word_vec = np.array(map(float,f.read().split("\n")))
        f.close()
        # this does not give equal weight to each year....
        # my_full_vec += my_word_vec
        # I guess, can't worry about the within-year warping now
        # this gives each year equal weight
        # (multiply by 1000 to keep the numbers reasonable)
        my_full_vec += (my_word_vec)/np.sum(my_word_vec)*1000

    for year in range(2008,2016):
        print("opening year {} to shift against bg".format(year))
        f = open("data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"r")
        my_word_vec = np.array(map(float,f.read().split("\n")))
        f.close()

        print("shifting")
        my_word_vec_stopped = my_senti_dict.stopper(my_word_vec,stopVal=stopVal)
        my_full_vec_stopped = my_senti_dict.stopper(my_full_vec,stopVal=stopVal)
        shiftHtml(my_senti_dict.scorelist, my_senti_dict.wordlist,
                 my_full_vec_stopped,my_word_vec_stopped,
                 "twitter-shift-{0:.0f}stop-{1}-{2:.0f}.html".format(stopVal*10,my_senti_dict.title,year),
                 # make_png_too=False,open_pdf=False,
                 customTitle=True,
                 # title="Twitter Wordshift using {0}, StopVal={1}".format(my_senti_dict.title,stopVal),
                 title="{0}: {1} Wordshift".format(letters[j],my_senti_dict.corpus),
                 ref_name="twitter all years combined",comp_name="twitter {:.0f}".format(year),)

def twitter_timeseries(start,end,resolution,my_senti_dict,stopVal):
    '''Load the full twitter timeseries from the word vectors.'''
    times = []
    timeseries = []

    base_resolution = timedelta(minutes=15)

    curr_date = start
    while curr_date < end:
        # now we need to add up all of the smaller resolutions files
        inner_date = curr_date
        my_word_vec = np.zeros(len(my_senti_dict.wordlist))
        # sum up as many as possible
        while inner_date < (curr_date+resolution):
            if my_senti_dict.corpus == "LabMT":
                 fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M.csv".format(word_vector_location))
            else:
                 fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M-{1}.csv".format(word_vector_location,my_senti_dict.corpus))
            # print(fname)
            if isfile(fname):
                f = open(fname,"r")
                tmp_word_vec = np.array(map(float,f.read().split("\n")))
                f.close()
                my_word_vec += tmp_word_vec
                if sum(tmp_word_vec) == 0 and error_logging:
                    f = open("empty-file-log-{0}.txt".format(my_senti_dict.corpus),"a")
                    f.write(fname)
                    f.write("\n")
                    f.close()  
            elif error_logging:
                f = open("missing-file-log-{0}.txt".format(my_senti_dict.corpus),"a")
                f.write(fname)
                f.write("\n")
                f.close()

            inner_date += base_resolution

        # compute the happs of my_word_vec
        my_word_vec = my_senti_dict.stopper(my_word_vec,stopVal=stopVal)
        if np.sum(my_word_vec) > 0:
            happs = np.dot(my_word_vec,my_senti_dict.scorelist)/np.sum(my_word_vec)
            timeseries.append(happs)
            times.append(curr_date)
        else:
            # print("found an empty one")
            # print(fname)
            # timeseries.append(-5.0)
            pass


        curr_date += resolution

    return times,timeseries

def twitter_timeseries_day_local(start,end,resolution,my_senti_dict,stopVal,return_total=False):
    '''Load the full twitter timeseries from the word vectors.
    Relies on the day -all- vector at the smallest resolution.'''
    
    times = []
    timeseries = []

    base_resolution = timedelta(days=1)

    curr_date = start
    my_total_word_vec = np.zeros(len(my_senti_dict.wordlist))
    # print(len(my_senti_dict.wordlist))
    while curr_date < end:
        # now we need to add up all of the smaller resolutions files
        inner_date = curr_date
        my_word_vec = np.zeros(len(my_senti_dict.wordlist))
        # sum up as many as possible
        while inner_date < (curr_date+resolution):
            fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-all-{1}.csv".format(word_vector_location,my_senti_dict.corpus))
            # print(fname)
            if isfile(fname):
                f = open(fname,"r")
                tmp_word_vec = np.array(map(float,f.read().split("\n")))
                f.close()
                try:
                    my_word_vec += tmp_word_vec
                    my_total_word_vec += tmp_word_vec
                except:
                    print(len(my_word_vec))
                    print(len(tmp_word_vec))
                    print(len(my_total_word_vec))
                    raise
                if sum(tmp_word_vec) == 0 and error_logging:
                    # print("empty one at {0}".format(fname))
                    f = open("logs/empty-file-log-{0}.txt".format(my_senti_dict.corpus),"a")
                    f.write(fname)
                    f.write("\n")
                    f.close()
            # no file, and in a reasonable date range
            elif error_logging and (inner_date > datetime(2008,9,15) or inner_date < datetime(2015,11,10)):
                # print("missing file at {0}".format(fname))
                f = open("logs/missing-file-log-{0}.txt".format(my_senti_dict.corpus),"a")
                f.write(fname)
                f.write("\n")
                f.close()

            inner_date += base_resolution

        # compute the happs of my_word_vec
        my_word_vec = my_senti_dict.stopper(my_word_vec,stopVal=stopVal)
        if np.sum(my_word_vec) > 0:
            happs = np.dot(my_word_vec,my_senti_dict.scorelist)/np.sum(my_word_vec)
            timeseries.append(happs)
            times.append(curr_date)
        else:
            # print("found an empty one")
            f = open("logs/zeros-file-log-{0}.txt".format(my_senti_dict.corpus),"a")
            f.write(fname)
            f.write("\n")
            f.close()  
            # pass
            
        curr_date += resolution
        
    if return_total:
        return times,timeseries,my_total_word_vec
    else:
        return times,timeseries

def twitter_timeseries_week(start,end,resolution,my_senti_dict,stopVal):
    """Make a full twitter timeseries with different resolutions, using word vecs."""

    base_resolution = timedelta(minutes=15)
    week = timedelta(weeks=1)
    day = timedelta(days=1)
    # a timeseries for each resolution, for each dictionary
    times = [i*resolution for i in range(int(week.total_seconds()/resolution.total_seconds()))]
    print(len(times))
    timeseries = np.zeros(len(times))
    wordvecs = [np.zeros(len(my_senti_dict.scorelist)) for i in range(len(times))]

    curr_date = start
    while curr_date < end:
        # now we need to add up all of the smaller resolutions files
        inner_date = curr_date
        my_word_vec = np.zeros(len(my_senti_dict.wordlist))
        # sum up as many as possible
        while inner_date < (curr_date+resolution):
            if my_senti_dict.corpus == "LabMT":
                 fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M.csv".format(word_vector_location))
            else:
                 fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M-{1}.csv".format(word_vector_location,my_senti_dict.corpus))
            # print(fname)
            if isfile(fname):
                f = open(fname,"r")
                # print(fname)
                tmp_word_vec = np.array(map(float,f.read().split("\n")))
                f.close()

                my_word_vec += tmp_word_vec

            inner_date += base_resolution

        # figure out the index in the week of the current date
        seconds_into_week = curr_date.weekday()*day.total_seconds()+curr_date.hour*3600+curr_date.minute*60
        index = int(np.round(seconds_into_week/week.total_seconds()*len(times)))
        # print(seconds_into_week)
        # print(index)
        wordvecs[index] += my_word_vec
        curr_date += resolution

    # compute the happs of my_word_vec
    for i,time in enumerate(times):
        my_word_vec = my_senti_dict.stopper(wordvecs[i],stopVal=stopVal)
        if np.sum(my_word_vec) > 0:
            happs = np.dot(my_word_vec,my_senti_dict.scorelist)/np.sum(my_word_vec)
        else:
            print("found an empty one")
            print(i)
        timeseries[i] = happs

    return (times,timeseries)

def twitter_timeseries_day(start,end,resolution,my_senti_dict,stopVal):
    """Make a full twitter timeseries with different resolutions, using word vecs."""

    base_resolution = timedelta(minutes=15)
    day = timedelta(days=1)
    # a timeseries for each resolution, for each dictionary
    times = [i*resolution for i in range(int(day.total_seconds()/resolution.total_seconds()))]
    print(len(times))
    timeseries = np.zeros(len(times))
    wordvecs = [np.zeros(len(my_senti_dict.scorelist)) for i in range(len(times))]

    curr_date = start
    while curr_date < end:
        # now we need to add up all of the smaller resolutions files
        inner_date = curr_date
        my_word_vec = np.zeros(len(my_senti_dict.wordlist))
        # sum up as many as possible
        while inner_date < (curr_date+resolution):
            if my_senti_dict.corpus == "LabMT":
                 fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M.csv".format(word_vector_location))
            else:
                 fname = inner_date.strftime("{0}%Y-%m-%d/%Y-%m-%d-%H-%M-{1}.csv".format(word_vector_location,my_senti_dict.corpus))
            # print(fname)
            if isfile(fname):
                f = open(fname,"r")
                # print(fname)
                tmp_word_vec = np.array(map(float,f.read().split("\n")))
                f.close()

                my_word_vec += tmp_word_vec

            inner_date += base_resolution

        # figure out the index in the week of the current date
        seconds_into_day = curr_date.hour*3600+curr_date.minute*60
        index = int(np.round(seconds_into_day/day.total_seconds()*len(times)))
        wordvecs[index] += my_word_vec
        curr_date += resolution

    # compute the happs of my_word_vec
    for i,time in enumerate(times):
        my_word_vec = my_senti_dict.stopper(wordvecs[i],stopVal=stopVal)
        if np.sum(my_word_vec) > 0:
            happs = np.dot(my_word_vec,my_senti_dict.scorelist)/np.sum(my_word_vec)
        else:
            print("found an empty one")
            print(i)
        timeseries[i] = happs

    return (times,timeseries)

def loadTwitter():
    """Load the full twitter corpus, and sort the total wordcounts."""

    # twitter_beginning = datetime(2015,7,1)
    # # twitter_end = datetime(2015,7,23)
    # twitter_end = datetime(2015,7,2)
    # allwordcounts = loadTweetsDynamic(twitter_beginning,twitter_end)
    # fname = "data/twitter/allwordcounts.dict"
    # pickle.dump( allwordcounts , open( fname , "wb" ) )

    fname = "data/twitter/allwordcounts.dict"
    allwordcounts = pickle.load( open( fname , "rb" ) )

    # allwordcounts = pickle.load( open("data/twitter/word-dicts/2015-07-01/2015-07-01-00-00.dict","rb") )

    print("there are {0} unique words in this corpus".format(str(len(allwordcounts))))

    # rip those dictionaries into lists for sorting
    allwordsList = [word for word in allwordcounts]
    allcountsList = [allwordcounts[allwordsList[i]] for i in range(len(allwordsList))]

    # sort them
    indexer = sorted(range(len(allcountsList)), key=lambda k: allcountsList[k], reverse=True)
    allcountsListSorted = np.array([float(allcountsList[i]) for i in indexer])
    allwordsListSorted = [allwordsList[i] for i in indexer]

    return allcountsListSorted,allwordsListSorted

def loadGBooks():
    """Load all of google books, which has already been combined for conveinence."""

    fname = "data/googlebooks/years/all-years.pickle"
    allwordcounts = pickle.load( open( fname , "rb" ) )

    print("there are {0} unique words in this corpus".format(str(len(allwordcounts))))

    # rip those dictionaries into lists for sorting
    allwordsList = [word for word in allwordcounts]
    allcountsList = [allwordcounts[allwordsList[i]] for i in range(len(allwordsList))]

    # sort them
    indexer = sorted(range(len(allcountsList)), key=lambda k: allcountsList[k], reverse=True)
    allcountsListSorted = np.array([float(allcountsList[i]) for i in indexer])
    allwordsListSorted = [allwordsList[i] for i in indexer]

    return allcountsListSorted,allwordsListSorted

def pickleNYT():
    # this little section turns them all into dicts, and sums them up
    sections = ["arts","books","classified","cultural","editorial","education","financial","foreign","home","leisure","living","magazine","metropolitan","movies","national","regional","science","society","sports","style","television","travel","week-in-review","weekend",]

    allwordcounts = dict()
    for section in sections:
        fname = "data/nyt/sections/NYT_{}.txt".format(section)
        print(fname)
        f = open(fname,"r")
        raw_text = f.read()
        f.close()
        section_dict = dict()
        dictify_general(raw_text,section_dict)
        fname = "data/nyt/sections/NYT_{}.dict".format(section)
        pickle.dump( section_dict , open( fname , "wb" ) )
        for word in section_dict:
            if word in allwordcounts:
                allwordcounts[word] += section_dict[word]
            else:
                allwordcounts[word] = section_dict[word]

    fname = "data/nyt/all.dict"
    pickle.dump( allwordcounts , open( fname , "wb" ) )

def loadNYT():
    """Load all of NYT, which has already been combined for convienence."""
    
    if not isfile("data/nyt/all.dict"):
        pickleNYT()
    
    fname = "data/nyt/all.dict"
    allwordcounts = pickle.load( open( fname , "rb" ) )

    print("there are {0} unique  words in this corpus".format(str(len(allwordcounts))))

    # rip those dictionaries into lists for sorting
    allwordsList = [word for word in allwordcounts]
    allcountsList = [allwordcounts[allwordsList[i]] for i in range(len(allwordsList))]

    # sort them
    indexer = sorted(range(len(allcountsList)), key=lambda k: allcountsList[k], reverse=True)
    allcountsListSorted = np.array([float(allcountsList[i]) for i in indexer])
    allwordsListSorted = [allwordsList[i] for i in indexer]

    return allcountsListSorted,allwordsListSorted

def loadtxt():
    allwordcounts = dict()
    # g = sys.stdin
    g = codecs.getreader("utf-8")(sys.stdin)
    for line in g:
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+",line,flags=re.UNICODE)]
        for word in words:
            if word in allwordcounts:
                allwordcounts[word] += 1
            else:
                allwordcounts[word] = 1

    print("there are {0} unique words in this corpus".format(str(len(allwordcounts))))

    # rip those dictionaries into lists for sorting
    allwordsList = [word for word in allwordcounts]
    allcountsList = [allwordcounts[allwordsList[i]] for i in range(len(allwordsList))]

    # sort them
    indexer = sorted(range(len(allcountsList)), key=lambda k: allcountsList[k], reverse=True)
    allcountsListSorted = np.array([float(allcountsList[i]) for i in indexer])
    allwordsListSorted = [allwordsList[i] for i in indexer]

    return allcountsListSorted,allwordsListSorted

def make_coverage_plot(allcountsListSorted,allwordsListSorted,corpus_title):
    titles = ["LabMT","ANEW","LIWC","MPQA","Liu","WK"]
    maxCount = 15000
    # maxCount = len(allcountsListSorted)
    total = np.sum(allcountsListSorted[:maxCount])

    def coverageMaker(wordList,sentimentTrie):
        a = np.array([float(sentimentTrie.matcherTrieBool(word)) for word in wordList[:maxCount]])
        b = np.cumsum(a)/(np.array(range(len(a)))+1)
        return a,b

    def totalCoverage(indices):
        return indices*allcountsListSorted[:maxCount]

    def covS(indices):
        return np.sum(totalCoverage(indices))/total

    def relativeCoverage(indices):
        totalCov = totalCoverage(indices)
        return np.cumsum(totalCov)/np.cumsum(allcountsListSorted[:maxCount])

    # make them all as both dicts and tries, with no stopval
    stopVal = 0.0
    LabMT_trie = LabMT(stopVal=stopVal)

    LIWC_trie = LIWC(stopVal=stopVal)
    WK_trie = WK(stopVal=stopVal)
    ANEW_trie = ANEW(stopVal=stopVal)
    MPQA_trie = MPQA(stopVal=stopVal)
    Liu_trie = Liu(stopVal=stopVal)

    labMTcoverage,labMTcovP = coverageMaker(allwordsListSorted,LabMT_trie)
    ANEWcoverage,ANEWcovP = coverageMaker(allwordsListSorted,ANEW_trie)
    LIWCcoverage,LIWCcovP = coverageMaker(allwordsListSorted,LIWC_trie)
    MPQAcoverage,MPQAcovP = coverageMaker(allwordsListSorted,MPQA_trie)
    liucoverage,liucovP = coverageMaker(allwordsListSorted,Liu_trie)
    WKcoverage,WKcovP = coverageMaker(allwordsListSorted,WK_trie)

    allCoverage = [labMTcoverage,ANEWcoverage,LIWCcoverage,MPQAcoverage,liucoverage,WKcoverage]
    allCovP = [labMTcovP,ANEWcovP,LIWCcovP,MPQAcovP,liucovP,WKcovP]
    allCovPfinal = [labMTcovP[-1],ANEWcovP[-1],LIWCcovP[-1],MPQAcovP[-1],liucovP[-1],WKcovP[-1]]

    save_individual_plots = False
    if save_individual_plots:
        plt.figure(num=None, figsize=(14, 9), dpi=600, facecolor="w", edgecolor="k")

        for i in range(len(allCovP)):
            plt.plot(range(maxCount),allCovP[i],linewidth=2)

        plt.xlabel("Word Rank")
        plt.ylabel("Percentage of individual words covered")

        plt.legend(titles,loc="best")
        mysavefig("word-coverage-by-rank-{0}.pdf".format(corpus_title),folder="figures/coverage")
        # mysavefig("word-coverage-by-rank-{0}.png".format(corpus_title))
        plt.close()

        coveragesBySet = list(map(covS,allCoverage))

        fig, ax = plt.subplots()
        # ax.bar(range(5),coveragesBySet,0.6,color=["r","b","g","k","c"])

        ax.bar(np.arange(len(allCoverage))+0.3,coveragesBySet,0.3,color="#ef8a62",)
        ax.bar(np.arange(len(allCoverage)),allCovPfinal,0.3,color="#2b8cbe",)
        ax.set_ylabel("Percentage")
        ax.set_title("Percentage coverage over first "+str(maxCount)+" words")
        # plt.legend(np.flipud(["Total Coverage","Individual Word Coverage"]),loc="best")
        plt.legend(["Total Coverage","Individual Word Coverage"],loc="best")
        ax.set_xlim([-.15,len(titles)-.3])
        ax.set_xticks(np.arange(len(allCoverage))+.3)
        ax.set_xticklabels( titles )
        ax.set_ylim([0,1])
        mysavefig("total-coverage-bar-chart-{0}.pdf".format(corpus_title),folder="figures/coverage")
        # mysavefig("total-coverage-bar-chart-{0}.png".format(corpus_title))
        plt.close()

        coveragesBySet2 = list(map(relativeCoverage,allCoverage))

        plt.figure(num=None, figsize=(14, 9), dpi=600, facecolor="w", edgecolor="k")
        for i in range(len(coveragesBySet2)):
            plt.plot(range(maxCount),coveragesBySet2[i],linewidth=2)
        plt.xlabel("Word Rank")
        plt.ylabel("Percentage of total words covered")
        plt.legend(titles,loc="best")
        mysavefig("relative-coverage-over-words-by-rank-{0}.pdf".format(corpus_title),folder="figures/coverage")
        # mysavefig("relative-coverage-over-words-by-rank-{0}.png".format(corpus_title))
        plt.close()
    # endif 

    # now the full subplot figure

    plt.figure(num=None, figsize=(figwidth_onecol, figwidth_onecol*.35), dpi=600, facecolor="w", edgecolor="k")
    ax = plt.subplot(131)

    for i in range(len(allCovP)):
        ax.plot(range(maxCount),allCovP[i],linewidth=2)

    ax.set_xlabel("Word Rank",fontsize=12)
    ax.set_ylabel("Percentage of individual words covered",fontsize=12)
    ax.set_ylim([0,1])
    ax.set_yticks([0,.2,.4,.6,.8,1.0])
    # ax.legend(titles,loc="best",fontsize=10)
    ax.set_xlim([0,maxCount])
    ax.set_xticks([0,5000,10000,15000])

    coveragesBySet2 = list(map(relativeCoverage,allCoverage))

    ax = plt.subplot(132)

    for i in range(len(coveragesBySet2)):
        ax.plot(range(maxCount),coveragesBySet2[i],linewidth=2)
    ax.set_xlabel("Word Rank",fontsize=12)
    ax.set_ylabel("Percentage of total words covered",fontsize=12)
    ax.legend(titles,loc="best",fontsize=10,ncol=2,framealpha=0.5)
    ax.set_ylim([0,1])
    ax.set_yticks([0,.2,.4,.6,.8,1.0])
    ax.set_xlim([0,maxCount])
    ax.set_xticks([0,5000,10000,15000])

    coveragesBySet = list(map(covS,allCoverage))

    ax = plt.subplot(133)

    ax.bar(np.arange(len(allCoverage))+0.3,coveragesBySet,0.3,color="#ef8a62",)
    ax.bar(np.arange(len(allCoverage)),allCovPfinal,0.3,color="#2b8cbe",)
    ax.set_ylabel("Percentage",fontsize=12)
    # ax.set_title("Percentage coverage over first "+str(maxCount)+" words")
    # ax.legend(np.flipud(["Total Coverage","Individual Word Coverage"]),loc="best")
    ax.legend(["Total Coverage","Individual Word\nCoverage"],loc="best",fontsize=10,framealpha=0.5)
    ax.set_xlim([-.15,len(titles)-.3])
    ax.set_xticks( np.arange( len( allCoverage ) ) +.3 )
    ax.set_xticklabels( titles , fontsize=12 , rotation=45 )
    ax.set_ylim( [0,1] )
    ax.set_yticks( [0,.2,.4,.6,.8,1.0] )

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.5)

    mysavefig("coverage-{0}.pdf".format(corpus_title),folder="figures/coverage")
    plt.close()

def sampleReviewsDict(numReviews,numSamples,filelist,wordsRE,prefix,test="LabMT-ANEW-LIWC-MPQA-Liu-WK"):
    """Sample from all of the review."""

    if numReviews == 1:
        choose_randomly = False
    else:
        choose_randomly = True

    scores = [[0.0 for i in range(numSamples)] for j in range(len(wordsRE))]
    for i in range(numSamples):
        # print("on sample {0}".format(i))

        if choose_randomly:
            files = np.random.choice(filelist,size=numReviews,replace=False)
        else:
            files = [filelist[i]]

        # forget the string expansion
        # let"s store them as a dict
        allwordcounts = dict()
        for file in files:
            # ########################################
            # # this makes the dicts if they're needed
            # if isfile(file+".dict"):
            #     my_dict = pickle.load( open( file+".dict", "rb" ) )
            #     for word in my_dict:
            #         if word in allwordcounts:
            #             allwordcounts[word] += my_dict[word]
            #         else:
            #             allwordcounts[word] = my_dict[word]
            # else:
            #     # read the txt file
            #     f = open(file+".txt","r")
            #     rawtext = f.read()
            #     f.close()
            #     # dictify_general it
            #     tmp_dict = dict()
            #     dictify_general(rawtext,tmp_dict)
            #     pickle.dump( tmp_dict , open( file+".dict", "wb" ) )
            #     # add to the full dict
            #     dictify_general(rawtext,allwordcounts)

            # ###################################################
            # # this loads the dicts
            # my_dict = pickle.load( open( file+".dict", "rb" ) )
            # for word in my_dict:
            #     if word in allwordcounts:
            #         allwordcounts[word] += my_dict[word]
            #     else:
            #         allwordcounts[word] = my_dict[word]

            ########################################
            # this loads the files
            f = open(file+".txt","r")
            rawtext = f.read()
            f.close()
            # add to the full dict
            dictify_general(rawtext,allwordcounts)

        for j in range(len(wordsRE)):
            scores[j][i] = wordsRE[j].score(allwordcounts)

    f = open("output/{0}/{1}-{2:.0f}-{3:.0f}.csv".format(test,prefix,numReviews,numSamples),"w")
    csv_writer = csv.writer(f)
    for row in scores:
        csv_writer.writerow(row)
    f.close()

def make_movie_review_plots(titles,allLengths,allSamples,save_individual_each_distribution=False,save_individual_overview=False,save_overview=True,save_comparison_separate=False,prefix="plot"):
    
    poscolor = "#ef8a62" # orange
    negcolor = "#2b8cbe" # blue
    
    negcolor = "#4C4CFF" # shift blue
    poscolor = "#FFFF4C" # shift yellow

    poscolor = "#ff7f00" # orange (shift yellow too hard to read)

    # read in all the data, take the mean and std and save them
    # these are titles x lengths
    posMeans = np.zeros([len(titles),len(allLengths),])
    posStds = np.zeros([len(titles),len(allLengths),])
    negMeans = np.zeros([len(titles),len(allLengths),])
    negStds = np.zeros([len(titles),len(allLengths),])
    overlapping = np.zeros([len(titles),len(allLengths),])

    for i in range(len(allLengths)):
        f = open("output/{0}/posScores-{1}-{2}.csv".format(prefix,allLengths[i],allSamples[i]),"r")
        csv_reader = csv.reader(f)
        posscores = [list(map(float,row)) for row in csv_reader]
        f.close()
        means = [np.mean(row) for row in posscores]
        stds = [np.std(row) for row in posscores]
        for j in range(len(titles)):
            posMeans[j,i] = means[j]
            posStds[j,i] = stds[j]
        f = open("output/{0}/negScores-{1}-{2}.csv".format(prefix,allLengths[i],allSamples[i]),"r")
        csv_reader = csv.reader(f)
        negscores = [list(map(float,row)) for row in csv_reader]
        f.close()

        # this will save out the distribution for each dictionary (for each review)
        if save_individual_each_distribution:
            for j in range(len(titles)):
                if i==0:
                    nbins = 50
                else:
                    nbins = 25

                fig = plt.figure()
                ax1 = fig.add_axes([0.15,0.2,0.7,0.7])
                ax1.hist(posscores[j], nbins, alpha=0.75, facecolor="#ef8a62",)
                ax1.hist(negscores[j], nbins, alpha=0.75, facecolor="#2b8cbe",)
                ax1.legend(["Positive reviews","Negative reviews"],loc="best",framealpha=0.5)
                # ax1.set_title("{0} score distribution for {1} reviews".format(titles[j],allLengths[i]))
                ax1.set_xlabel("Score")
                ax1.set_ylabel("Number of reviews")

                # mysavefig("{0}-{1}reviews-dist".format(titles[j],allLengths[i]))
                # mysavefig("{0}-{1}reviews-dist".format(titles[j],allLengths[i]))
                mysavefig("reviews-dist-{0}-{1}.pdf".format(titles[j],allLengths[i]),date=False,folder="figures/moviereviews")
                plt.close(fig)

        means = [np.mean(row) for row in negscores]
        stds = [np.std(row) for row in negscores]
        for j in range(len(titles)):
            negMeans[j,i] = means[j]
            negStds[j,i] = stds[j]

        # now go and compute the overlap
        for j in range(len(titles)):
            # #######################################################
            # # this compute the full overlap (with a failed attempt
            # # at computing only part of it
            # conf = .95
            # negmax = np.max(negscores[j])
            # # negmax = sorted(negscores[j])[-1]
            # # negmax = sorted(negscores[j])[int(np.rint(conf*len(negscores[j])))-1]
            # posmin = np.min(posscores[j])
            # # posmin = sorted(posscores[j])[0]
            # # posmin = sorted(posscores[j])[int(np.rint((1-conf)*len(posscores[j])))+1]
            # if posmin < negmax:
            #     # overlapRange = np.array([posmin,negmax])
            #     overlapping[j,i] = float(len(np.where(posscores[j] < negmax)[0]) + len(np.where(negscores[j] > posmin)[0])) /(len(posscores[j])+len(negscores[j]))

            # #########################################################
            # # now just how many cross the center
            # center = (posMeans[j,i] + negMeans[j,i]) / 2
            # overlapping[j,i] = float(len(np.where(posscores[j] < center)[0]) + len(np.where(negscores[j] > center)[0])) /(len(posscores[j])+len(negscores[j]))

            #########################################################
            # now just how many cross the average
            average_score = np.mean(negscores[j]+posscores[j])
            overlapping[j,i] = float(len(np.where(posscores[j] < average_score)[0]) + len(np.where(negscores[j] > average_score)[0])) /(len(posscores[j])+len(negscores[j]))

    if save_individual_overview:
        for j in range(len(titles)):
            fig = plt.figure()
            ax1 = fig.add_axes([0.15,0.2,0.7,0.7])
            ax1.errorbar(np.log10(allLengths),posMeans[j],posStds[j],linewidth=2,color="#ef8a62",)
            ax1.errorbar(np.log10(allLengths),negMeans[j],negStds[j],linewidth=2,color="#2b8cbe",)
            ax1.legend(["Positive reviews","Negative reviews"],loc="best",framealpha=0.5)
            ax1.set_xlabel("log10(num reviews)")
            ax1.set_ylabel("sentiment")
            ax1.set_xlim(np.log10([allLengths[0]-.1,allLengths[-1]+.1])) 
            # ax1.set_ylim([0,24])
            # plt.xticks([float(i)+0.5 for i in range(4)])
            # plt.yticks([float(i)+0.5 for i in range(3)])
            # ax1.set_xticklabels([1,5,25,50])
            # ax1.set_yticklabels([22,28,35])
            ax1.set_title("sentiment averaged over 100 random samples for {0}".format(titles[j]))
            mysavefig("{0}-sentimentVsSize.pdf".format(titles[j]),folder="figures/moviereviews")
            plt.close(fig)

    if save_overview:
        panels = ["A","B","C","D","E","F","G","G"]
        
        fig = plt.figure(num=None, figsize=(figwidth_onecol,figwidth_onecol*.7), facecolor="w", edgecolor="k")
        # fig.set_tight_layout(True)
        panel_i = 0
        # whole figure label padding
        xpad = .085
        ypad = .085
        xpadr = .015
        ypadr = .015
        # individual tick padding
        xtickpad = .035
        ytickpad = .035
        # remaining width
        xrem = 1.-xpad-xpadr
        yrem = 1.-ypad-ypadr
        # divide it up
        plots_x = 4
        plots_y = 2
        xwidth = xrem/(plots_x+1)
        ywidth = xrem/(plots_y+1)
        for j in range(len(titles)):
            # ax = plt.subplot(3,5,j+1)
            rect = [xpad+np.mod(j,plots_x)*xwidth+xtickpad,ypad+(plots_y-1-np.floor(j/plots_x))*ywidth,xwidth-xtickpad,ywidth-ytickpad]
            ax = fig.add_axes(rect)
            # ax = plt.subplot(2,4,j+1)
            ax.errorbar(np.log10(allLengths),posMeans[j],posStds[j],linewidth=1,color=poscolor,)
            ax.errorbar(np.log10(allLengths),negMeans[j],negStds[j],linewidth=1,color=negcolor,)
            ax.legend(["Pos. reviews","Neg. reviews"],loc="upper right",fontsize=8,framealpha=0.5)
            # ax.set_title(titles[j])
            # these convert it to number of words
            # ax.set_xticks(np.log10p(allLengths))
            # ax.set_xticks(np.log10(np.logspace(0,3,num=5)))
            # ax.set_xticklabels(np.log10([x*650 for x in allLengths]))
            # ax.set_xticklabels(list(map(lambda x: "{0:.0f}".format(x),[x*650 for x in np.logspace(0,3,num=5)])))
            if j == 0 or j == 4:
                ax.set_ylabel("Sentiment",fontsize=8)
            ax.set_xlim(np.log10([allLengths[0]-.1,allLengths[-1]+.1]))
            if j > 3:
                # ax.set_xlabel("log10(num words)")
                ax.set_xlabel(r"$\log_{10}$(Num. Reviews)",fontsize=8)
                ax.set_xticklabels([0.0,0.5,1.0,1.5,2.0,2.5])
            else:
                ax.set_xticklabels([])
            wheat = "#F5DEB3"
            beige = "#F5F5DC"
            bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec ="k", lw=1)
            # ax.text(np.log10(allLengths[-1])-.25, np.min(ax.get_yticks()), "{0}".format(panels[panel_i]),ha="right", va="bottom", fontsize=14, bbox=bbox_props)
            ax.text( 0.91, 0.05,"{0}: {1}".format(panels[panel_i],titles[j]) , transform=ax.transAxes, ha="right", va="bottom", fontsize=10, bbox=bbox_props)
            panel_i+=1

        # ax = plt.subplot(2,2,4)
        j = 6
        rect = [xpad+np.mod(j,plots_x)*xwidth+xtickpad/2,ypad+(plots_y-1-np.floor(j/plots_x))*ywidth,xwidth*2-xtickpad,ywidth-ytickpad]
        ax = fig.add_axes(rect)        
        # ax = plt.subplot(3,2,6)
        print("-"*80)
        print("results from the movie reviews")
        print(allLengths)
        for j in range(len(titles)):
            print(titles[j])
            print(1-overlapping[j,:])
            ax.plot(np.log10(allLengths),overlapping[j,:],linewidth=1)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("\% Score overlap",fontsize=8)
        # ax.set_xticks(np.log10(allLengths))
        # ax.set_xticks(np.log10(np.logspace(0,3,num=5)))
        # # ax.set_xticklabels(list(map(lambda x: "{0:.0f}".format(x),np.log10([x*650 for x in allLengths]))))
        # ax.set_xticklabels(list(map(lambda x: "{0:.0f}".format(x),[x*650 for x in np.logspace(0,3,num=5)])))
        # ax.set_xlabel("log10(num reviews)")
        # ax.set_xlabel("log10(num words)")
        ax.set_xlabel("$\log_{10}$(Num. Reviews)",fontsize=8)
        ax.text( 0.04, 0.05,"{0}: All".format(panels[panel_i]) , transform=ax.transAxes, ha="left", va="bottom", fontsize=10, bbox=bbox_props)
        plt.legend(titles,fontsize=8)
        # mysavefig("movie-review-overview-{0}.png".format(prefix))
        # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.5)
        mysavefig("movie-review-overview-{0}.pdf".format(prefix),folder="figures/moviereviews")
        # plt.show()

    if save_comparison_separate:
        fig = plt.figure()
        ax = fig.add_axes([0.15,0.2,0.7,0.7])
        for j in range(len(titles)):
            ax.plot(np.log10(allLengths),overlapping[j,:],linewidth=1)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Fraction of overlapping scores")
        # ax.set_xticks(np.log10(allLengths))
        ax.set_xticks(np.log10(np.logspace(0,3,num=5)))
        # ax.set_xticklabels(list(map(lambda x: "{0:.0f}".format(x),np.log10([x*650 for x in allLengths]))))
        ax.set_xticklabels(list(map(lambda x: "{0:.0f}".format(x),[x*650 for x in np.logspace(0,3,num=5)])))
        # ax.set_xlabel("log10(num reviews)")
        # ax.set_xlabel("log10(num words)")
        ax.set_xlabel("log10(Number of Reviews)")
        plt.legend(titles)
        mysavefig("movie-review-comparison.pdf".format(prefix),folder="figures/moviereviews")
        # mysavefig("movie-review-comparison.png".format(prefix),folder="figures/moviereviews")
        # plt.show()

def coverage():
    """Make each of the four main coverage plots."""

    corpus = "twitter"
    print("making coverage plot for {0}".format(corpus))
    allcountsListSorted,allwordsListSorted = loadTwitter()
    make_coverage_plot(allcountsListSorted,allwordsListSorted,corpus)

    corpus = "movieReviews"
    print("making coverage plot for {0}".format(corpus))
    allcountsListSorted,allwordsListSorted = loadMovieReviews()
    make_coverage_plot(allcountsListSorted,allwordsListSorted,corpus)

    corpus = "googleBooks"
    print("making coverage plot for {0}".format(corpus))
    allcountsListSorted,allwordsListSorted = loadGBooks()
    make_coverage_plot(allcountsListSorted,allwordsListSorted,corpus)

    corpus = "nyt"
    print("making coverage plot for {0}".format(corpus))
    allcountsListSorted,allwordsListSorted = loadNYT()
    make_coverage_plot(allcountsListSorted,allwordsListSorted,corpus)


def reviewTest(test,allLengths,allSamples):
    """Do the full review test. Takes 20 minutes."""

    stopVal = 0.0
    allDicts = (ANEW(stopVal=stopVal),
                LIWC(stopVal=stopVal),
                MPQA(stopVal=stopVal),
                Liu(stopVal=stopVal),
                WK(stopVal=stopVal))    

    for flip in ["pos","neg"]:
        files = ["data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
                 for x in listdir("data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
        prefix = "{0}Scores".format(flip)

        # numReviews = int(sys.argv[3])
        # # if we have all of the reviews, only one possible sample
        # if numReviews>(len(files)-10):
        #     numSamples = 1
        # else:
        #     numSamples = int(sys.argv[4])

        # sampleReviewsDict(numReviews,numSamples,files,allDicts,prefix)

        for i,numReviews in enumerate(allLengths):
            numSamples = allSamples[i]
            print("taking {0} samples of {1} reviews".format(numSamples,numReviews))
            sampleReviewsDict(numReviews,numSamples,files,allDicts,prefix,test=test)

# global allDicts
# global all_stopVals

def gbook_timeseries(allDicts,save_shifts=True,use_cache=True):
    """Make a full gbook timeseries, using word dicts."""

    all_stopVals = [0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5,0.5,0.5,0.5]
    all_order = [0,0,0,1,1,1,2,2,2,3,4,5]

    years = sorted([1900,1905,1910,1915,1920,1925,1930,1935,1940,1945,1950,1955,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,1901,1906,1911,1916,1921,1926,1931,1936,1941,1946,1951,1956,1961,1966,1971,1976,1981,1986,1991,1996,2001,2006,1902,1907,1912,1917,1922,1927,1932,1937,1942,1947,1952,1957,1962,1967,1972,1977,1982,1987,1992,1997,2002,2007,1903,1908,1913,1918,1923,1928,1933,1938,1943,1948,1953,1958,1963,1968,1973,1978,1983,1988,1993,1998,2003,2008,1904,1909,1914,1919,1924,1929,1934,1939,1944,1949,1954,1959,1964,1969,1974,1979,1984,1989,1994,1999,2004])

    start = datetime(years[0],1,1)
    end = datetime(years[-1]+1,1,1)
    # base_resolution = timedelta(years=1)

    # a timeseries for each resolution, for each dictionary
    all_timeseries = [[] for j in range(len(allDicts))]
    times = []

    # loop over all dictionaries
    for j,my_senti_dict in enumerate(allDicts):
        print("dictionary {0}".format(j))
        curr_date = start
        total_vec = np.zeros(len(my_senti_dict.scorelist))
        while curr_date < end:
            # print(curr_date)
            # read teh pickle, word vec it

            my_word_vec = np.zeros(len(my_senti_dict.scorelist))

            csvfile = curr_date.strftime("data/googlebooks/years/%Y/all-100k-{0}.csv".format(my_senti_dict.corpus))
            if isfile(csvfile) and use_cache:
                f = open(csvfile,"r")
                my_word_vec = np.array(map(float,f.read().split("\n")))
                f.close()
            else:
                year_word_dict = pickle.load( open( curr_date.strftime("data/googlebooks/years/%Y/all-100k.pickle") , "rb" ) )
                my_word_vec = my_senti_dict.wordVecify(year_word_dict)
                # write out the word vecs
                f = open(csvfile,"w")
                f.write("\n".join(list(map(lambda x: "{0:.0f}".format(x),my_word_vec))))
                f.close()

            total_vec += my_word_vec

            # print(max(my_word_vec))
            # compute the happs of my_word_vec
            my_word_vec = my_senti_dict.stopper(my_word_vec,stopVal=all_stopVals[j])
            happs = np.dot(my_word_vec,my_senti_dict.scorelist)/np.sum(my_word_vec)
            all_timeseries[j].append(happs)

            if j == 0:
                times.append(curr_date)

            # curr_date += resolution
            # hack to increment year
            curr_date = curr_date.replace(year=curr_date.year+1)

        total_vec_stopped = my_senti_dict.stopper(total_vec,stopVal=all_stopVals[j])

        print("shift loop {0}".format(j))
        if save_shifts:
            curr_date = start
            while curr_date < end:
                print(curr_date)
                # read teh pickle, word vec it

                my_word_vec = np.zeros(len(my_senti_dict.scorelist))

                inner_date = curr_date
                while inner_date < curr_date.replace(year=curr_date.year+10):
                    csvfile = inner_date.strftime("data/googlebooks/years/%Y/all-100k-{0}.csv".format(my_senti_dict.corpus))
                    picklefile = inner_date.strftime("data/googlebooks/years/%Y/all-100k.pickle")
                    print(csvfile)
                    if isfile(csvfile):
                        f = open(csvfile,"r")
                        my_word_vec += np.array(map(float,f.read().split("\n")))
                        f.close()
                    elif isfile(picklefile):
                        year_word_dict = pickle.load( open( picklefile , "rb" ) )
                        my_word_vec += my_senti_dict.wordVecify(year_word_dict)
                        # write out the word vecs
                        f = open(csvfile,"w")
                        f.write("\n".join(list(map(lambda x: "{0:.0f}".format(x),my_word_vec))))
                        f.close()
                    inner_date = inner_date.replace(year=inner_date.year+1)

                # print(max(my_word_vec))
                # compute the happs of my_word_vec
                my_word_vec_stopped = my_senti_dict.stopper(my_word_vec,stopVal=all_stopVals[j])

                shiftHtml(my_senti_dict.scorelist, my_senti_dict.wordlist,
                    total_vec_stopped, my_word_vec_stopped,
                    "GBooks-shift-{0}-{1:.1f}-{2}-decade.html".format(my_senti_dict.corpus,all_stopVals[j],curr_date.strftime("%Y")),
                    # make_png_too=False,open_pdf=False,
                    customTitle=True,
                    title="{0}: {1} Wordshift".format(letters[all_order[j]],my_senti_dict.corpus),
                    ref_name="Google Books as a whole",comp_name=curr_date.strftime("%Y's"),
                    ref_name_happs="Google Books as a whole",comp_name_happs=curr_date.strftime("%Y's"))

                # curr_date += resolution
                # hack to increment year
                curr_date = curr_date.replace(year=curr_date.year+10)

    return (times,all_timeseries)

def plot_gbook_timeseries():
    stopVal = 0.0
    allDicts = (LabMT(stopVal=stopVal),
                LabMT(stopVal=stopVal),
                LabMT(stopVal=stopVal),
                ANEW(stopVal=stopVal),
                ANEW(stopVal=stopVal),
                ANEW(stopVal=stopVal),
                WK(stopVal=stopVal),
                WK(stopVal=stopVal),
                WK(stopVal=stopVal),
                MPQA(stopVal=stopVal),
                LIWC(stopVal=stopVal),
                Liu(stopVal=stopVal),)

    times,all_timeseries = gbook_timeseries(allDicts,save_shifts=False)
    # print(times)
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_axes([0.2,0.2,0.6,0.7])

    # for j in [1,4,7,9,10,11]: # range(len(allDicts)):
    #     # ax.set_title("Google Books by Year")
    #     np_timeseries = np.array(all_timeseries[j])
    #     # print(np_timeseries)
    #     norm_timeseries = (np_timeseries-np_timeseries.mean())/(np_timeseries.max()-np_timeseries.min())
    #     ax.plot(times,norm_timeseries)
    #     ax.set_ylabel("Normalized Sentiment",fontsize=12)
    #     ax.set_xlabel("Years",fontsize=12)
    #     # ax.plot(times[i],all_timeseries[i][j])
    #     # ax.plot(times[i],(np_timeseries-np_timeseries.mean())/(np_timeseries.max()-np_timeseries.min()),"-",markersize=5,linewidth=1.0)
    # ax.legend([my_dict.corpus for my_dict in [allDicts[i] for i in [1,4,7,9,10,11]]],loc="center left", bbox_to_anchor=(1, 0.5))
    # # mysavefig("gbooks-timeseries-{0}-normalized.png".format("6"))
    # # mysavefig("gbooks-timeseries-{0}-normalized.pdf".format("6"))

    labels = ["LabMT 0.5","LabMT 1.0","LabMT 1.5",
              "ANEW 0.5","ANEW 1.0","ANEW 1.5",
              "WK 0.5","WK 1.0","WK 1.5",
              "LIWC","MPQA","Liu"]

    color = ["#000000","#000000","#000000",
             "#FF5050","#FF5050","#FF5050",
             "#3366FF","#3366FF","#3366FF",
             "#00CC00","#FF33CC","#CC9900",]

    linestyle = ["-","--",":",
                 "-","--",":",
                 "-","--",":",
                 "-","-","-",]

    rc("xtick", labelsize=15)
    rc("ytick", labelsize=15)

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_axes([0.2,0.2,0.6,0.7])

    for j in range(len(allDicts)):
        # ax.set_title("Google Books by Year")
        np_timeseries = np.array(all_timeseries[j])
        norm_timeseries = (np_timeseries-np_timeseries.mean())/(np_timeseries.max()-np_timeseries.min())
        ax.plot(times,norm_timeseries,color=color[j],label=labels[j],linewidth=1.5,alpha=0.9,linestyle=linestyle[j])
        ax.set_ylabel("Normalized Sentiment",fontsize=22)
        ax.set_xlabel("Years",fontsize=22)
        # ax.plot(times[i],all_timeseries[i][j])
        # ax.plot(times[i],(np_timeseries-np_timeseries.mean())/(np_timeseries.max()-np_timeseries.min()),"-",markersize=5,linewidth=1.0)
    years = sorted([1900,1905,1910,1915,1920,1925,1930,1935,1940,1945,1950,1955,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,1901,1906,1911,1916,1921,1926,1931,1936,1941,1946,1951,1956,1961,1966,1971,1976,1981,1986,1991,1996,2001,2006,1902,1907,1912,1917,1922,1927,1932,1937,1942,1947,1952,1957,1962,1967,1972,1977,1982,1987,1992,1997,2002,2007,1903,1908,1913,1918,1923,1928,1933,1938,1943,1948,1953,1958,1963,1968,1973,1978,1983,1988,1993,1998,2003,2008,1904,1909,1914,1919,1924,1929,1934,1939,1944,1949,1954,1959,1964,1969,1974,1979,1984,1989,1994,1999,2004])
    tickyears = [10,20,30,40,50,60,70,80,90,100,]
    tickyearslabels = [years[i] for i in tickyears]
    tickyears = [datetime(x,1,1) for x in tickyearslabels]
    ax.set_xticks(tickyears)
    ax.set_xticklabels(tickyearslabels)
    # ax.set_xticks(years)
    # ax.legend([my_dict.corpus for my_dict in [allDicts[i] for i in [1,4,7,9,10,11]]],loc="best")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),fontsize=15)
    # mysavefig("gbooks-timeseries-{0}-normalized.png".format("all"))
    mysavefig("gbooks-timeseries-{0}-normalized.pdf".format("all"),folder="figures/googlebooks")

    num_dicts = len(allDicts)
    correlations = np.zeros((num_dicts,num_dicts))
    for i in range(num_dicts):
        for j in range(num_dicts):
            cor = pearsonr(all_timeseries[i],all_timeseries[j])
            # print(cor)
            correlations[i,j] = cor[0]

    # fig = plt.figure(figsize=(12,10))
    fig = plt.figure(figsize=(figwidth_onecol,figwidth_onecol*.8))
    # fig = plt.figure(figsize=(figwidth_twocol,figwidth_twocol*.80))
    ax = fig.add_axes([.2,.2,.7,.7])
    ax.set_yticks(np.arange(12)+0.5)
    ax.set_yticklabels(labels,fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(12)+0.5)
    ax.set_xticklabels(labels,rotation=45,fontsize=10) #,offset=0.3)
    cs = ax.pcolor(correlations.transpose(),cmap=plt.get_cmap("RdYlBu"))
    plt.colorbar(cs,ax=ax,shrink=0.9)
    # mysavefig("gbooks-correlations-years-diverging.png")
    mysavefig("gbooks-correlations-years-diverging.pdf",folder="figures/googlebooks")

def combine_word_lists(senti_dict_1,senti_dict_2):
    combined_words = []
    # indices_1 = np.zeros(len(senti_dict_1.wordlist))
    # indices_2 = np.zeros(len(senti_dict_2.wordlist))
    indices_1 = [x for x in range(len(senti_dict_1.wordlist))]
    indices_2 = [-1 for x in range(len(senti_dict_2.wordlist))]
    i = 0
    j = 0
    k = 0
    # march through both lists
    while i<len(senti_dict_1.wordlist) or j<len(senti_dict_2.wordlist):
        # if any less from list 1
        if i<len(senti_dict_1.wordlist):
            combined_words.append(senti_dict_1.wordlist[i])
            indices_1[i] = k
            # check if it's in list 2
            if senti_dict_1.wordlist[i] in senti_dict_2.wordlist:
                indices_2[senti_dict_2.wordlist.index(senti_dict_1.wordlist[i])] = k
            i+=1
        else:
            if indices_2[j] == -1:
                combined_words.append(senti_dict_2.wordlist[j])
                indices_2[j] = k
            else:
                k-=1
            j+=1
        k+=1
    return (combined_words,indices_1,indices_2)

def testDualShift():
    LabMT_trie = LabMT(stopVal=stopVal)
    ANEW_trie = ANEW(stopVal=stopVal)
    WK_trie = WK(stopVal=stopVal)
    MPQA_trie = MPQA(stopVal=stopVal)
    LIWC_trie = LIWC(stopVal=stopVal)
    Liu_trie = Liu(stopVal=stopVal)

    combined_words,indices_1,indices_2 = combine_word_lists(LabMT_trie,Liu_trie)

    # print(combined_words,indices_1,indices_2)

    pos_wordcounts,neg_wordcounts = loadMovieReviewsBoth()

    pos_word_vec_labMT = LabMT_trie.wordVecify(pos_wordcounts)
    pos_word_vec_liu = Liu_trie.wordVecify(pos_wordcounts)
    pos_word_vec_stopped_labMT = LabMT_trie.stopper(pos_word_vec_labMT,stopVal=1.0)
    pos_word_vec_stopped_liu = Liu_trie.stopper(pos_word_vec_liu,stopVal=0.5)
    pos_word_vec_stopped_labMT_combined = np.zeros(len(combined_words))
    scorelist_labMT_combined = np.zeros(len(combined_words))
    for i,k in enumerate(indices_1):
        pos_word_vec_stopped_labMT_combined[k] = pos_word_vec_stopped_labMT[i]
        scorelist_labMT_combined[k] = (LabMT_trie.scorelist[i]-4.9)/3.6
    pos_word_vec_stopped_liu_combined = np.zeros(len(combined_words))
    scorelist_liu_combined = np.zeros(len(combined_words))
    for i,k in enumerate(indices_2):
        pos_word_vec_stopped_liu_combined[k] = pos_word_vec_stopped_liu[i]
        scorelist_liu_combined[k] = Liu_trie.scorelist[i]
    shiftHtmlDual(scorelist_labMT_combined, combined_words,
                  pos_word_vec_stopped_labMT_combined, pos_word_vec_stopped_liu_combined,
                  scorelist_liu_combined,
                  "movie-shift-Liu-vs-LabMT.html",
                  customTitle=True,
                  title="All Positive Movie Reviews: LabMT vs Liu",
                  ref_name="LabMT positive reviews",comp_name="Liu positive reviews",
                  ref_name_happs="LabMT positive reviews",comp_name_happs="Liu positive reviews",)



    neg_word_vec_labMT = LabMT_trie.wordVecify(neg_wordcounts)
    neg_word_vec_liu = Liu_trie.wordVecify(neg_wordcounts)
    neg_word_vec_stopped_labMT = LabMT_trie.stopper(neg_word_vec_labMT,stopVal=2.5)
    neg_word_vec_stopped_liu = Liu_trie.stopper(neg_word_vec_liu,stopVal=0.5)
    neg_word_vec_stopped_labMT_combined = np.zeros(len(combined_words))
    scorelist_labMT_combined = np.zeros(len(combined_words))
    for i,k in enumerate(indices_1):
        neg_word_vec_stopped_labMT_combined[k] += neg_word_vec_stopped_labMT[i]
        scorelist_labMT_combined[k] = (LabMT_trie.scorelist[i]-4.9)/3.6
    neg_word_vec_stopped_liu_combined = np.zeros(len(combined_words))
    scorelist_liu_combined = np.zeros(len(combined_words))
    for i,k in enumerate(indices_2):
        neg_word_vec_stopped_liu_combined[k] += neg_word_vec_stopped_liu[i]
        scorelist_liu_combined[k] = Liu_trie.scorelist[i]
    shiftHtmlDual(scorelist_labMT_combined, combined_words,
             neg_word_vec_stopped_labMT_combined, neg_word_vec_stopped_liu_combined,
             scorelist_liu_combined,
             "movie-shift-Liu-vs-LabMT-negative-wide.html",
                     customTitle=True,
                     title="All Negative Movie Reviews: LabMT vs Liu",
                     ref_name="LabMT negative reviews",comp_name="Liu negative reviews",
                     ref_name_happs="LabMT negative reviews",comp_name_happs="Liu negative reviews",)


def plot_twitter_timeseries():
    """From the files saved in output/, will plot the twitter timeseries for
    3 different resolutions (biggest 3)."""
    resolutions = [timedelta(minutes=15),timedelta(hours=1),timedelta(hours=3),timedelta(hours=12),timedelta(days=1),]
    resolution_titles = ["15 Minutes","1 Hour","3 Hours","12 Hours","1 Day",]
    resolution_widths = [1920,480,160,40,40,]
    
    # make them all as both dicts and tries, with no stopval
    stopVal = 0.0
    LabMT_dict = LabMT()
    LabMT_trie = LabMT()
    LIWC_dict = LIWC(stopVal=0.0,bananas=False)
    LIWC_trie = LIWC(bananas=False)
    WK_dict = WK()
    WK_trie = WK()
    ANEW_dict = ANEW()
    ANEW_trie = ANEW()
    MPQA_dict = MPQA()
    MPQA_trie = MPQA()
    Liu_dict = Liu()
    Liu_trie = Liu()
    
    # allDicts = (LabMT_dict,
    #             ANEW_dict,
    #             WK_dict,
    #             LIWC_trie,
    #             MPQA_trie,
    #             Liu_trie,)

    allDicts = (LabMT_dict,
                LabMT_dict,
                LabMT_dict,
                ANEW_dict,
                ANEW_dict,
                ANEW_dict,
                WK_dict,
                WK_dict,
                WK_dict,
                LIWC_trie,
                MPQA_trie,
                Liu_trie,)
    all_stopVals = [0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5,0.5,0.5,0.5]
    
    labels = ["LabMT 0.5","LabMT 1.0","LabMT 1.5",
              "ANEW 0.5","ANEW 1.0","ANEW 1.5",
              "WK 0.5","WK 1.0","WK 1.5",
              "LIWC","MPQA","Liu"]
    
    color = ["#000000","#000000","#000000",
             "#FF5050","#FF5050","#FF5050",
             "#3366FF","#3366FF","#3366FF",
             "#00CC00","#FF33CC","#CC9900",]
    
    linestyle = ["-","--",":",
                 "-","--",":",
                 "-","--",":",
                 "-","-","-",]

    rc("xtick", labelsize=15)
    rc("ytick", labelsize=15)
    
    # for i in [2,3,4]: # range(5):
    for i in [4]:
        resolution = resolutions[i]
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_axes([0.2,0.2,0.6,0.7])
        for j in range(len(allDicts)):
            stopVal = all_stopVals[j]
            my_senti_dict = allDicts[j]
            print("opening output/twitter-timeseries-{0}-{1}-{2}.csv".format(i,my_senti_dict.corpus,stopVal))
            f = open("output/twitter-timeseries-{0}-{1}-{2}.csv".format(i,my_senti_dict.corpus,stopVal),"r")
            lines = f.read().split("\n")
            f.close()
            f = open("output/twitter-timeseries-{0}-{1}-{2}-times.csv".format(i,my_senti_dict.corpus,stopVal),"r")
            times =list(map(lambda x: datetime.strptime(x,"%Y-%m-%d-%H-%M"),f.read().split("\n")))
            f.close()
            # print(lines)
            happs = np.array(map(float,lines))
            happs_norm = (happs-happs.mean())/(happs.max()-happs.min())
            # times = [datetime(2015,8,4) + k*resolution for k in range(int(day.total_seconds()/resolution.total_seconds()))]
            # ax.plot(times,happs_norm,label=(my_senti_dict.corpus),alpha=1.0,linewidth=0.5)
            ax.plot(times,happs_norm,color=color[j],label=labels[j],linewidth=1.0,alpha=1.0,linestyle=linestyle[j])
            # ax.set_xlabel("Time")
            # ax.set_ylabel("Happs")
            # ax.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            # ax.set_title("Daily happiness timeseries from {0} with stopVal {1} at resolution of {2}".format(my_senti_dict.corpus,stopVal,resolution_titles[i]))
            # mysavefig("twitter-timeseries-day-{0}-{1}-{2}.pdf".format(my_senti_dict.corpus,stopVal,resolution_titles[i].replace(" ","-").lower()))
            # plt.close(fig)
        ax.set_ylabel("Normalized Sentiment",fontsize=22)
        ax.set_xlabel("Years",fontsize=22)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),fontsize=15)
        mysavefig("twitter-timeseries-all-{0}.pdf".format(resolution_titles[i].replace(" ","-").lower()))
        # plt.close(fig)

def plot_twitter_correlations():
    """Probably still buried in the python notebook to work correctly,
    but this should get most of the job done to plot the correlation matrices for
    all resolutions."""

    resolutions = [timedelta(minutes=15),timedelta(hours=1),timedelta(hours=3),timedelta(hours=12),timedelta(days=1),]
    resolution_titles = ["15 Minutes","1 Hour","3 Hours","12 Hours","1 Day",]
    allDicts = (LabMT(),
                ANEW(),
                WK(),
                LIWC(stopVal=0.5),
                MPQA(),
                Liu(),)
    
    # 15 minute and 1 hour resolution have some sligtly different times
    # goal: make a times subset, then use it the next time
    mutual_times = [[] for i in range(5)]
    for i,resolution in enumerate(resolutions):
        for j in [0,1,2]:
            my_senti_dict = allDicts[j]
            for stopVal in [0.5,1.0,1.5,2.0]:
                f = open("output/twitter-timeseries-{0}-{1}-{2}-times.csv".format(i,my_senti_dict.corpus,stopVal),"r")
                times = list(map(lambda x: datetime.strptime(x,"%Y-%m-%d-%H-%M"),f.read().split("\n")))
                f.close()
                if len(mutual_times[i]) == 0:
                    mutual_times[i] = times
                else:
                    print("combining mutual times")
                    mutual_times[i] = list(set(mutual_times[i]) & set(times))
        for j in [3,4,5]:
            stopVal = 0.5
            my_senti_dict = allDicts[j]
            f = open("output/twitter-timeseries-{0}-{1}-{2}-times.csv".format(i,my_senti_dict.corpus,stopVal),"r")
            times =list(map(lambda x: datetime.strptime(x,"%Y-%m-%d-%H-%M"),f.read().split("\n")))
            f.close()
            if len(mutual_times[i]) == 0:
                mutual_times[i] = times
            else:
                print("combinging mutual_times")
                mutual_times[i] = list(set(mutual_times[i]) & set(times))
            print(len(times))
            print(len(mutual_times[i]))
    # now go get the timeseries, and subset for the mutual times        
    all_timeseries = [[] for i in range(5)]
    for i in range(5):
        resolution = resolutions[i]
        for j in [0,1,2]:
            my_senti_dict = allDicts[j]
            for stopVal in [0.5,1.0,1.5,2.0]:
                print("opening output/twitter-timeseries-{0}-{1}-{2}.csv".format(i,my_senti_dict.corpus,stopVal))
                f = open("output/twitter-timeseries-{0}-{1}-{2}.csv".format(i,my_senti_dict.corpus,stopVal),"r")
                lines = f.read().split("\n")
                f.close()
                happs = np.array(map(float,lines))
                print("subsetting the happs")
                all_timeseries[i].append(happs[:len(mutual_times[i])])
        for j in [3,4,5]:
            stopVal = 0.5
            my_senti_dict = allDicts[j]
            print("opening output/twitter-timeseries-{0}-{1}-{2}.csv".format(i,my_senti_dict.corpus,stopVal))
            f = open("output/twitter-timeseries-{0}-{1}-{2}.csv".format(i,my_senti_dict.corpus,stopVal),"r")
            lines = f.read().split("\n")
            f.close()
            happs = np.array(map(float,lines))
            print("subsetting the happs")
            all_timeseries[i].append(happs[:len(mutual_times[i])])
                
    print(len(all_timeseries))
    print(len(all_timeseries[0]))

    # now go compute the correlations
    for resolution_i in range(5):
        correlations = np.zeros([len(all_timeseries[resolution_i]),len(all_timeseries[resolution_i])])
        for i in range(len(all_timeseries[resolution_i])):
            for j in range(len(all_timeseries[resolution_i])):
                cor = pearsonr(all_timeseries[resolution_i][i],all_timeseries[resolution_i][j])
                # print(cor)
                correlations[i,j] = cor[0]
    
        labels = ["LabMT 0.5","LabMT 1.0","LabMT 1.5","LabMT 2.0",
                  "ANEW 0.5","ANEW 1.0","ANEW 1.5","ANEW 2.0",
                  "WK 0.5","WK 1.0","WK 1.5","WK 2.0",
                  "LIWC","MPQA","Liu"]
    
        fig = plt.figure(figsize=(figwidth_onecol,figwidth_onecol*.7))
        ax = fig.add_axes([.2,.2,.7,.7])
        ax.set_yticks(np.arange(len(all_timeseries[resolution_i]))+0.5)
        ax.set_yticklabels(labels,fontsize=10)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(len(all_timeseries[resolution_i]))+0.5)
        ax.set_xticklabels(labels,rotation=45,fontsize=10) # ,offset=0.3)
        cs = ax.pcolor(correlations.transpose(),cmap=plt.get_cmap("RdYlBu"))
        plt.colorbar(cs,ax=ax,shrink=0.9)
        # mysavefig("twitter-correlations-all-{0}.png".format(resolution_i))
        mysavefig("twitter-correlations-all-{0}.pdf".format(resolution_i))

if __name__ == "__main__": 
    if sys.argv[1] == "coverage":
        coverage()

    if sys.argv[1] == "movie-review-test":
        # let these be global....
        # (variable for the movie review test)

        # datastructure = "marisatrie"
        # stopVal = 1.0
        # allDicts = (LabMT(datastructure=datastructure,stopVal=stopVal),
        #             ANEW(datastructure=datastructure,stopVal=stopVal),
        #             LIWC(datastructure=datastructure,stopVal=stopVal),
        #             MPQA(datastructure=datastructure,stopVal=stopVal),
        #             Liu(datastructure=datastructure,stopVal=stopVal),
        #             WK(datastructure=datastructure,stopVal=stopVal))

        # test = "LabMT-ANEW-LIWC-MPQA-Liu-WK-quick-test"
        # allLengths = [1,3,10,30,100,300,1000]
        # allSamples = [1000,100,100,100,100,100,1]

        # test = "LabMT-ANEW-LIWC-MPQA-Liu-WK"
        # allLengths = [1,2,3,5,7,10,15,25,40,60,80,100,150,250,400,600,900,1000]
        # allSamples = [100 for i in range(len(allLengths))]
        # allSamples[0] = 1000
        # allSamples[-1] = 1

        # test = "LabMT-ANEW-LIWC-MPQA-Liu-WK-dynamic-sample"
        # allLengths = [1,   2,  3,  5,  7,  10, 15,  25, 40, 60, 80,100,150,250,400,600,900,1000]
        # allSamples = [1000,750,600,500,400,300,200,200,200,100,100, 75, 75, 55, 40, 25, 15,   1]

        test = "LabMT-ANEW-LIWC-MPQA-Liu-WK-dynamic-bigger"
        allLengths = [1,   2,   3,   5,   7,   10, 15,  25, 40, 60, 80,100,150,250,400,600,900,1000]
        allSamples = [1000,2000,1500,1500,1000,900,750,600,500,250,100, 75, 75, 55, 40, 25, 15,   1]

        # # LabMT-stop-window
        # test = "LabMT-stop-window-test-more"
        # allDicts = (LabMT(stopVal=0.0),
        #             LabMT(stopVal=0.25),
        #             LabMT(stopVal=0.5),
        #             LabMT(stopVal=0.75),
        #             LabMT(stopVal=1.0),
        #             LabMT(stopVal=1.25),
        #             LabMT(stopVal=1.5),
        #             LabMT(stopVal=1.75),
        #             LabMT(stopVal=2.0),
        #             LabMT(stopVal=2.25),
        #             LabMT(stopVal=2.5),)
        # allLengths = [1,   2,  3,  5,  7,  10, 15,  25, 40, 60, 80,100,150,250,400,600,900,1000]
        # allSamples = [1000,100,100,100,100,100,100,100,100,100,100, 75, 75, 55, 40, 25, 15,   1]
        reviewTest(test,allLengths,allSamples)

    if sys.argv[1] == "plot-movie-reviews":
        titles = ["LabMT","ANEW","LIWC","MPQA","Liu","WK"]
        # titles = ["LabMT0","LabMT0.25","LabMT0.5","LabMT0.75","LabMT1.0","LabMT1.25","LabMT1.5","LabMT1.75","LabMT2.0","LabMT2.25","LabMT2.5"]

        test = "LabMT-ANEW-LIWC-MPQA-Liu-WK-dynamic-bigger"
        allLengths = [1,   2,   3,   5,   7,   10, 15,  25, 40, 60, 80,100,150,250,400,600,900,1000]
        allSamples = [1000,2000,1500,1500,1000,900,750,600,500,250,100, 75, 75, 55, 40, 25, 15,   1]
        make_movie_review_plots(titles,allLengths,allSamples,prefix=test)

    if sys.argv[1] == "review-wordshifts":
        make_movie_shifts()
        # call("./figures/moviereviews-shifts/combine.sh",shell=True)

    if sys.argv[1] == "plot-google-books":
        plot_gbook_timeseries()

    if sys.argv[1] == "plot-google-books-shifts":
        stopVal = 0.0
        allDicts = (LabMT(stopVal=stopVal),
                    LabMT(stopVal=stopVal),
                    LabMT(stopVal=stopVal),
                    ANEW(stopVal=stopVal),
                    ANEW(stopVal=stopVal),
                    ANEW(stopVal=stopVal),
                    WK(stopVal=stopVal),
                    WK(stopVal=stopVal),
                    WK(stopVal=stopVal),
                    MPQA(stopVal=stopVal),
                    LIWC(stopVal=stopVal),
                    Liu(stopVal=stopVal),)

        gbook_timeseries(allDicts,save_shifts=True)
        # call("./figures/googlebooks-shifts/combine.sh",shell=True)

    if sys.argv[1] == "twitter-addyears":
        stopVal = 0.0
        # twitter_addyears(LabMT(stopVal=stopVal))
        # twitter_addyears(ANEW(stopVal=stopVal))
        # twitter_addyears(WK(stopVal=stopVal))
        # twitter_addyears(LIWC(stopVal=stopVal))
        # twitter_addyears(MPQA(stopVal=stopVal))
        # twitter_addyears(Liu(stopVal=stopVal))
        twitter_addyears_local(LIWC(stopVal=stopVal))
        twitter_addyears_local(LabMT(stopVal=stopVal))
        twitter_addyears_local(ANEW(stopVal=stopVal))
        twitter_addyears_local(WK(stopVal=stopVal))
        twitter_addyears_local(MPQA(stopVal=stopVal))
        twitter_addyears_local(Liu(stopVal=stopVal))
        
    if sys.argv[1] == "twitter-correlations":
        plot_twitter_correlations()

    if sys.argv[1] == "twitter-shiftyears":
        stopVal = 1.0
        twitter_shiftyears(LabMT(stopVal=0.0),stopVal=stopVal,j=0)
        twitter_shiftyears(ANEW(stopVal=0.0),stopVal=stopVal,j=1)
        twitter_shiftyears(WK(stopVal=0.0),stopVal=stopVal,j=2)
        stopVal = 0.5
        twitter_shiftyears(MPQA(stopVal=0.0),stopVal=stopVal,j=3)
        twitter_shiftyears(LIWC(stopVal=0.0),stopVal=stopVal,j=4)
        twitter_shiftyears(Liu(stopVal=0.0),stopVal=stopVal,j=5)
        # call("./figures/twitter-shifts/combine.sh",shell=True)

    if sys.argv[1] == "twitter-plot-timeseries":
        plot_twitter_timeseries()

    if sys.argv[1] == "make-twitter-wordvecs-day":
        date = datetime.strptime(sys.argv[2],"%Y-%m-%d")
        make_twitter_wordvecs_day(date)

    if sys.argv[1] == "twitter-timeseries":
        # short test
        # twitter_beginning = datetime(2008,9,1)
        # twitter_end = datetime(2008,9,15)
        # full timeseries
        twitter_beginning = datetime(2008,9,12)
        # DATA GOES THROUGH 2015-04-10
        twitter_end = datetime(2015,4,10)
        
        # load from sys.argv
        # allwordcounts = make_twitter_wordvecs_full_res(twitter_beginning,twitter_end)

        # let's do this one at a time....
        # times,all_timeseries = twitter_timeseries_all(twitter_beginning,twitter_end)

        # straight from the _all function
        resolutions = [timedelta(minutes=15),timedelta(hours=1),timedelta(hours=3),timedelta(hours=12),timedelta(days=1),]

        resolution = resolutions[int(sys.argv[3])]

        stopVal = 0.0
        allDicts = (LabMT(stopVal=stopVal),
                    ANEW(stopVal=stopVal),
                    WK(stopVal=stopVal),
                    LIWC(stopVal=stopVal),
                    MPQA(stopVal=stopVal),
                    Liu(stopVal=stopVal),)

        my_senti_dict = allDicts[int(sys.argv[4])]

        stopVal = float(sys.argv[5])

        print("running")

        # times,timeseries = twitter_timeseries(twitter_beginning,twitter_end,resolution,my_senti_dict,stopVal)
        # times,timeseries = twitter_timeseries_week(twitter_beginning,twitter_end,resolution,my_senti_dict,stopVal)
        # times,timeseries = twitter_timeseries_day(twitter_beginning,twitter_end,resolution,my_senti_dict,stopVal)

        # save the timeseries
        f = open("output/twitter-timeseries-{0}-{1}-{2}-day.csv".format(sys.argv[3],my_senti_dict.corpus,sys.argv[5]),"w")
        f.write("\n".join(list(map(lambda x: "{0:.7f}".format(x),timeseries))))
        f.close()

        # f = open("output/twitter-timeseries-{0}-{1}-{2}-times-day.csv".format(sys.argv[3],my_senti_dict.corpus,sys.argv[5]),"w")
        # f.write("\n".join(list(map(lambda x: x.strftime("%Y-%m-%d-%H-%M"),times))))
        # f.close()

        # twitter_timeseries_week(twitter_beginning,twitter_end)
        # twitter_timeseries_day(twitter_beginning,twitter_end)
        # all_timeseries = twitter_timeseries(twitter_beginning,twitter_end)

        print("job completion success")

    if sys.argv[1] == "dual-shift":
        testDualShift()

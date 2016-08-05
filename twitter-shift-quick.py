# load the very latest version
import sys
sys.path.append("/Users/andyreagan/work/2014/03-labMTsimple/")
from labMTsimple.speedy import *
from labMTsimple.storyLab import *

sys.path.append("/Users/andyreagan/work/2015/08-kitchentabletools/")
from dog.toys import *

import numpy as np

my_senti_dict = LabMT()
my_full_vec = np.zeros(len(my_senti_dict.wordlist))
year = 2012
print("opening year {} to add it up".format(year))
f = open("data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"r")
my_word_vec_2012 = np.array(map(float,f.read().split("\n")))
f.close()

year = 2013
print("opening year {} to add it up".format(year))
f = open("data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"r")
my_word_vec_2013 = np.array(map(float,f.read().split("\n")))
f.close()

year = 2015
print("opening year {} to add it up".format(year))
f = open("data/twitter/{1}-{0}.csv".format(my_senti_dict.title,year),"r")
my_word_vec_2015 = np.array(map(float,f.read().split("\n")))
f.close()

stopVal=1.0
my_word_vec_2012_stopped = my_senti_dict.stopper(my_word_vec_2012,stopVal=stopVal)
my_word_vec_2013_stopped = my_senti_dict.stopper(my_word_vec_2013,stopVal=stopVal)
my_word_vec_2015_stopped = my_senti_dict.stopper(my_word_vec_2015,stopVal=stopVal)
shiftHtml(my_senti_dict.scorelist, my_senti_dict.wordlist,
          my_word_vec_2012_stopped,my_word_vec_2015_stopped,
          "twitter-shift-2015-vs-2012.html".format(stopVal*10,my_senti_dict.title),
          customTitle=True,
          title="Twitter Wordshift",
          ref_name="twitter 2012",comp_name="twitter 2015",)

shiftHtml(my_senti_dict.scorelist, my_senti_dict.wordlist,
          my_word_vec_2013_stopped,my_word_vec_2015_stopped,
          "twitter-shift-2015-vs-2013.html".format(stopVal*10,my_senti_dict.title),
          customTitle=True,
          title="Twitter Wordshift",
          ref_name="twitter 2013",comp_name="twitter 2015",)


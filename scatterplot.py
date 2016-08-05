import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc("font",**{"family":"serif","serif":["cmr10"]})
rc('text', usetex=True)

rc("xtick", labelsize=10)
rc("ytick", labelsize=10)

import scipy.odr.odrpack as odrpack
import numpy as np
np.random.seed(1)

# load the very latest version
import sys
sys.path.append("/Users/andyreagan/work/2014/03-labMTsimple/")
from labMTsimple.speedy import *
from labMTsimple.storyLab import *

sys.path.append("/Users/andyreagan/work/2015/08-kitchentabletools/")
# from kitchentable.tools import *
# better namespacing
from dog.toys import *

LabMT_dict = LabMT()
LabMT_trie = LabMT(datastructure='marisatrie',stopVal=0.0)
LIWC_dict = LIWC(stopVal=0.0,bananas=False)
LIWC_trie = LIWC(datastructure='marisatrie',stopVal=0.0,bananas=False)
WK_dict = WK()
ANEW_dict = ANEW()
MPQA_dict = MPQA()
MPQA_trie = MPQA(datastructure='marisatrie')
Liu_dict = Liu()
Liu_trie = Liu(datastructure='marisatrie')

my_LabMT = LabMT(stopVal=0.0)
my_LIWC = LIWC(stopVal=0.0)
my_WK = WK(stopVal=0.0)
my_ANEW = ANEW(stopVal=0.0)
my_MPQA = MPQA(stopVal=0.0)
my_Liu = Liu(stopVal=0.0,datastructure='marisatrie')

def make_shift_plots_all():
    dataList = [LabMT_dict,ANEW_dict,WK_dict,MPQA_trie,LIWC_trie,Liu_dict]

    # down (this is the y in the plots)
    for i in range(1,len(dataList)):
        # across (this is the x in the plots)
        for j in range(0,i):
            print(i,j)
            print(dataList[j].title,dataList[i].title)
            print(dataList[j].score_range,dataList[i].score_range)
            if dataList[j].score_range == 'full' and dataList[i].score_range == 'integer':
                makeShiftPlotsFullInteger(dataList[j],dataList[i])
            if dataList[j].score_range == 'full' and dataList[i].score_range == 'full':
                makeShiftPlotsFullFull(dataList[j],dataList[i])

def makeShiftPlotsFullFull(x_set,y_set):
    fname = '{0}-{1}'.format(x_set.title,y_set.title)
    overlapWords = []
    overlapScores = []
    overlapScoresStd = []
    # since 
    for word,index in x_set.data.items():
        if word in y_set.data:
            overlapScores.append((index[0],y_set.data[word][0]))
            overlapScoresStd.append((index[2],y_set.data[word][2]))
            overlapWords.append(word)

    x_setscores = [x[0] for x in overlapScores]
    x_setscoresStd = [x[0] for x in overlapScoresStd]
    y_setscores = [x[1] for x in overlapScores]
    y_setscoresStd = [x[1] for x in overlapScoresStd]
    
    # ordinary least squares
    # mydata = odrpack.RealData(x_setscores, y_setscores) #, sx=sx, sy=sy)
    # RMA
    mydata = odrpack.RealData(x_setscores, y_setscores, sx=x_setscoresStd, sy=y_setscoresStd)

    myodr = odrpack.ODR(mydata, linear, beta0=[1., 0.])
    myoutput = myodr.run()
    myoutput.pprint()
    # it fit to this:
    # myoutput.beta[0]*x+myoutput.beta[1]
    beta = myoutput.beta

    # let's now go and compute the distance for each word
    absDiff = [np.abs(x[1]-x[0]) for x in overlapScores]

    fitDiff = np.zeros(len(overlapScores))
    for i,scores in zip(range(len(overlapScores)),overlapScores):
        intercept_x = (scores[1]+scores[0]*beta[0]-beta[1])/(2*beta[0])
        intercept_y = beta[0]*intercept_x + beta[1]
        d = np.sqrt((intercept_x-scores[0])**2+(intercept_y-scores[1])**2)
        fitDiff[i] = d

    indexer = sorted(range(len(overlapScores)), key=lambda k: fitDiff[k], reverse=True)
    overlapScoresSorted = [overlapScores[i] for i in indexer]
    overlapWordsSorted = [overlapWords[i] for i in indexer]
    fitDiffSorted = [fitDiff[i] for i in indexer]
    print('sorted by difference from the fit:')
    print(overlapScoresSorted[:10])
    print(overlapWordsSorted[:10])
    print(fitDiffSorted[:10])

    f = open('{0}-sorted-overlapping-words-fit.txt'.format(fname),'w')
    f.write('word,score1,score2,diff\n')
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,fitDiffSorted):
        f.write('{0},{1:.2f},{2:.2f},{3:.2f}\n'.format(word,scores[0],scores[1],fitdiff))
    f.close()

    print('sorted by absolute difference:')
    indexer = sorted(range(len(overlapScores)), key=lambda k: absDiff[k], reverse=True)
    overlapScoresSorted = [overlapScores[i] for i in indexer]
    overlapWordsSorted = [overlapWords[i] for i in indexer]
    absDiffSorted = [absDiff[i] for i in indexer]
    print(overlapScoresSorted[:10])
    print(overlapWordsSorted[:10])
    print(absDiffSorted[:10])

    f = open('{0}-sorted-overlapping-words-abs.txt'.format(fname),'w')
    f.write('word,score1,score2,diff\n')
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,absDiffSorted):
        f.write('{0},{1:.2f},{2:.2f},{3:.2f}\n'.format(word,scores[0],scores[1],fitdiff))
    f.close()

def makeShiftPlotsFullInteger(x_set,y_set):
    fname = '{0}-{1}'.format(x_set.title,y_set.title)
    # this code is from scatter_full_integer...but I don't need to draw figures
    overlapWords = []
    overlapScores = []
    for word,index in x_set.data.items():
        if word in y_set.data[0]:
            score = y_set.data[0].get(word)[0][0]
            overlapWords.append(word)
            overlapScores.append((index[0],score))
        elif len(y_set.data[1].prefixes(word)) > 0:
            score = y_set.data[1].get(y_set.data[1].prefixes(word)[0])[0][0]
            overlapWords.append(word)
            overlapScores.append((index[0],score))
    overlapScoresSplit = [[],[],[]]
    overlapWordsSplit = [[],[],[]]
    for i,word in zip(range(len(overlapWords)),overlapWords):
        if overlapScores[i][1] == -1:
            overlapScoresSplit[0].append(overlapScores[i][0])
            overlapWordsSplit[0].append(word)
        elif overlapScores[i][1] == 1:
            overlapScoresSplit[2].append(overlapScores[i][0])
            overlapWordsSplit[2].append(word)
        else:
            overlapScoresSplit[1].append(overlapScores[i][0])
            overlapWordsSplit[1].append(word)

    fvec = x_set.wordVecify(dictify(overlapWordsSplit[0]))
    shiftHtmlSelf(x_set.fixedscores,x_set.fixedwords,fvec,"{0}-negwords.html".format(fname))

    fvec = x_set.wordVecify(dictify(overlapWordsSplit[1]))
    shiftHtmlSelf(x_set.fixedscores,x_set.fixedwords,fvec,"{0}-neutralwords.html".format(fname))

    fvec = x_set.wordVecify(dictify(overlapWordsSplit[2]))
    shiftHtmlSelf(x_set.fixedscores,x_set.fixedwords,fvec,"{0}-poswords.html".format(fname))
    
def scatter_full_full(x_set,y_set,ax,insetsize=6,tile="A"):
    '''...'''
    overlapWords = []
    overlapScores = []
    overlapScoresStd = []
    for word,index in y_set.data.items():
        if word in x_set.data:
            overlapScores.append((x_set.data[word][1],index[1]))
            overlapScoresStd.append((x_set.data[word][2],index[2]))
            overlapWords.append(word)
    # print(len(overlapWords))
    # print(len(overlapScores))
    # print(len(overlapScoresStd))
    # print(overlapScores[:10])
    # print(overlapScoresStd[:10])
    # print(overlapWords[:10])

    x_setscores = [x[0] for x in overlapScores]
    x_setscoresStd = [x[0] for x in overlapScoresStd]
    # print('median of set 1 is: {0}'.format(np.median(x_setscores)))
    y_setscores = [x[1] for x in overlapScores]
    # print('median of set 2 is: {0}'.format(np.median(y_setscores)))
    # print('which makes the difference: {0}'.format(np.abs(np.median(x_setscores)-np.median(y_setscores))))
    y_setscoresStd = [x[1] for x in overlapScoresStd]
    
    # ordinary least squares
    # mydata = odrpack.RealData(x_setscores, y_setscores) #, sx=sx, sy=sy)
    # RMA
    mydata = odrpack.RealData(x_setscores, y_setscores, sx=x_setscoresStd, sy=y_setscoresStd)

    myodr = odrpack.ODR(mydata, linear, beta0=[1., 0.])
    myoutput = myodr.run()
    myoutput.pprint()
    # print(myoutput.beta)
    # print(myoutput.sd_beta)
    beta = myoutput.beta    
    
    # ax.scatter(x_setscores,y_setscores,alpha=0.25,marker='o',s=6,linewidth=0.2,edgecolor='k')
    ax.scatter(x_setscores,y_setscores,alpha=0.9,marker='o',c='#F0F0FA',s=12,linewidth=0.0,edgecolor='k')
    ax.scatter(x_setscores,y_setscores,alpha=0.9,marker='o',c='#4D4D4D',s=0.7,linewidth=0.0,edgecolor='k')
    
    x = np.linspace(min(x_setscores),max(x_setscores),num=100)
    ax.plot(x,myoutput.beta[0]*x+myoutput.beta[1],'r',linewidth=0.75)
    
    ax.legend(['$\\beta$ = {0:.2f}\n$\\alpha$ = {1:.2f}'.format(myoutput.beta[0],myoutput.beta[1])],loc='upperleft',fontsize=insetsize,frameon=False)

    # ax1.set_xlabel('LabMT',fontsize=14)
    # ax1.set_ylabel('WK',fontsize=14)
    ax.set_xlim([1,9])
    ax.set_ylim([1,9])
    ax.set_xticks([])
    ax.set_yticks([])
    # mysavefig('WK-LabMT.pdf')

    # let's now go and compute the distance for each word
    absDiff = [np.abs(x[1]-x[0]) for x in overlapScores]

    fitDiff = np.zeros(len(overlapScores))
    for i,scores in zip(range(len(overlapScores)),overlapScores):
        intercept_x = (scores[1]+scores[0]*beta[0]-beta[1])/(2*beta[0])
        intercept_y = beta[0]*intercept_x + beta[1]
        d = np.sqrt((intercept_x-scores[0])**2+(intercept_y-scores[1])**2)
        fitDiff[i] = d

    indexer = sorted(range(len(overlapScores)), key=lambda k: fitDiff[k], reverse=True)
    overlapScoresSorted = [overlapScores[i] for i in indexer]
    overlapWordsSorted = [overlapWords[i] for i in indexer]
    fitDiffSorted = [fitDiff[i] for i in indexer]
    print('sorted by difference from the fit:')
    print(overlapScoresSorted[:10])
    print(overlapWordsSorted[:10])
    print(fitDiffSorted[:10])

    f = open('txt-output/{0}-{1}-sorted-overlapping-words-fit.csv'.format(x_set.title,y_set.title),'w')
    f.write('word,score1,score2,diff\n')
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,fitDiffSorted):
        f.write('{0},{1:.2f},{2:.2f},{3:.2f}\n'.format(word,scores[0],scores[1],fitdiff))
    f.close()

    f = open('{0}-{1}-sorted-overlapping-words-fit.tex'.format(x_set.title,y_set.title),'w')
    f.write('\\begin{tabular}[t]{p{1.5cm}|p{1.0cm}|p{1.0cm}|p{1.0cm}}\n')
    f.write('Word & $h_{{\\text{{ {0} }} }}$ & $h_{{ \\text{{ {1} }} }}$ & $h_{{\\text{{diff}}}}$ \\\\\n\\hline\n'.format(x_set.title,y_set.title))
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,fitDiffSorted)[:10]:
        f.write('{0} & \\centering\\arraybackslash {1:.2f} & \\centering\\arraybackslash {2:.2f} & \\centering\\arraybackslash {3:.2f}\\\\\n'.format(word,scores[0],scores[1],fitdiff))
    f.write('\\end{tabular}\n')
    f.close()

    tabletex_file_to_pdf('{0}-{1}-sorted-overlapping-words-fit.tex'.format(x_set.title,y_set.title))
    f = open('{0}.tex'.format(tile),'w')
    f.write('\\begin{tabular}[t]{p{1.5cm}|p{1.0cm}|p{1.0cm}|p{1.0cm}}\n')
    f.write('Word & $h_{{\\text{{ {0} }} }}$ & $h_{{ \\text{{ {1} }} }}$ & $h_{{\\text{{diff}}}}$ \\\\\n\\hline\n'.format(x_set.title,y_set.title))
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,fitDiffSorted)[:10]:
        f.write('{0} & \\centering\\arraybackslash {1:.2f} & \\centering\\arraybackslash {2:.2f} & \\centering\\arraybackslash {3:.2f}\\\\\n'.format(word,scores[0],scores[1],fitdiff))
    f.write('\\end{tabular}\n')
    f.close()
    tabletex_file_to_pdf('{0}.tex'.format(tile))

    print('sorted by absolute difference:')
    indexer = sorted(range(len(overlapScores)), key=lambda k: absDiff[k], reverse=True)
    overlapScoresSorted = [overlapScores[i] for i in indexer]
    overlapWordsSorted = [overlapWords[i] for i in indexer]
    absDiffSorted = [absDiff[i] for i in indexer]
    print(overlapScoresSorted[:10])
    print(overlapWordsSorted[:10])
    print(absDiffSorted[:10])

    f = open('txt-output/{0}-{1}-sorted-overlapping-words-abs.csv'.format(x_set.title,y_set.title),'w')
    f.write('word,score1,score2,diff\n')    
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,absDiffSorted):
        f.write('{0},{1:.2f},{2:.2f},{3:.2f}\n'.format(word,scores[0],scores[1],fitdiff))
    f.close()

    f = open('{0}-{1}-sorted-overlapping-words-abs.tex'.format(x_set.title,y_set.title),'w')
    f.write('\\begin{tabular}[t]{p{1.5cm}|p{1.0cm}|p{1.0cm}|p{1.0cm}}\n')
    f.write('Word & $h_{{\\text{{ {0} }} }}$ & $h_{{ \\text{{ {1} }} }}$ & $h_{{\\text{{diff}}}}$ \\\\\n\\hline\n'.format(x_set.title,y_set.title))    
    for scores,word,fitdiff in zip(overlapScoresSorted,overlapWordsSorted,fitDiffSorted)[:10]:
        f.write('{0} & \\centering\\arraybackslash {1:.2f} & \\centering\\arraybackslash {2:.2f} & \\centering\\arraybackslash {3:.2f}\\\\\n'.format(word,scores[0],scores[1],fitdiff))
    f.write('\\end{tabular}\n')            
    f.close()

    tabletex_file_to_pdf('{0}-{1}-sorted-overlapping-words-abs.tex'.format(x_set.title,y_set.title))

def histogram_WK_LabMT(x_set,y_set,ax,insetsize=6):
    xmin = 1
    xmax = 9
    step = .2

    # let's plot them as two histograms, without the bars
    overlapWords = []
    overlapScores = []
    overlapScoresStd = []
    for word,index in x_set.items():
        if word in y_set:
            overlapScores.append((index[0],y_set[word][0]))
            overlapScoresStd.append((index[2],y_set[word][2]))
            overlapWords.append(word)
    print(len(overlapWords))
    print(len(overlapScores))
    print(len(overlapScoresStd))
    print(overlapScores[:10])
    print(overlapScoresStd[:10])
    print(overlapWords[:10])

    x_setscores = [x[0] for x in overlapScores]
    y_setscores = [x[1] for x in overlapScores]
    
    ax.set_xlim([1,9])
    # ax.set_ylim([1,9])
    ax.set_xticks([])
    ax.yaxis.tick_right()
    # ax.set_yticks([])

    ax.set_ylabel('count')
    ax.set_xlabel('happs')

    y1, x1 = np.histogram(x_setscores, bins=np.linspace(xmin, xmax, (xmax-xmin)/step))
    ax.plot(x1[:-1]+step/2,y1,'b-8')

    y2, x2 = np.histogram(y_setscores, bins=np.linspace(xmin, xmax, (xmax-xmin)/step))
    ax.plot(x2[:-1]+step/2,y2,'r-^')

    ax.legend(['WK','LabMT'],fontsize=insetsize,loc='best')
    # mysavefig('WK-LabMT.pdf')

def f(B, x):
    return B[0]*x + B[1]

linear = odrpack.Model(f)

def scatter_integer_integer_3(x_set,y_set,axlist,tile="A"):
    overlapWords = []
    overlapScores = []
    # capture the num of stem matches
    overlapWordsStem = []
    overlapScoresStem = []
    for word,index in y_set.my_marisa[0].items():
        # exact fixed matches
        if word in x_set.my_marisa[0]:
            score = x_set.my_marisa[0].get(word)[0][1]
            overlapWords.append(word)
            overlapScores.append((score,index[1]))
        # this uses x_set.my_marisa's stems to search y_set.my_marisa's fixed words
        elif len(x_set.my_marisa[1].prefixes(word)) > 0:
            score = x_set.my_marisa[1].get(x_set.my_marisa[1].prefixes(word)[0])[0][1]
            overlapWordsStem.append(word)
            overlapScoresStem.append((score,index[1]))
    for word,index in y_set.my_marisa[1].items():
        # exact stem matches
        if word in x_set.my_marisa[1]:
            score = x_set.my_marisa[1].get(word)[0][1]
            overlapWords.append(word)
            overlapScores.append((score,index[1]))
        # this allows x_set.my_marisa's stems to match y_set.my_marisa's stemmed words
        elif len(x_set.my_marisa[1].prefixes(word)) > 0:
            score = x_set.my_marisa[1].get(x_set.my_marisa[1].prefixes(word)[0])[0][1]
            overlapWordsStem.append(word)
            overlapScoresStem.append((score,index[1]))
    print(len(overlapWords))
    print(len(overlapScores))
    print(overlapScores[:10])
    print(overlapWords[:10])
    print(len(overlapWordsStem))
    print(len(overlapScoresStem))
    print(overlapScoresStem[:10])
    print(overlapWordsStem[:10])

    # these are going to split onto the -1,0,1 of the y set
    overlapScoresSplit = [[],[],[]]
    overlapWordsSplit = [[],[],[]]
    for i,word in zip(range(len(overlapWords)),overlapWords):
        if overlapScores[i][1] == -1:
            overlapScoresSplit[0].append(overlapScores[i][0])
            overlapWordsSplit[0].append(word)
        elif overlapScores[i][1] == 1:
            overlapScoresSplit[2].append(overlapScores[i][0])
            overlapWordsSplit[2].append(word)
        else:
            overlapScoresSplit[1].append(overlapScores[i][0])
            overlapWordsSplit[1].append(word)
    overlapScoresSplitStem = [[],[],[]]
    overlapWordsSplitStem = [[],[],[]]
    for i,word in zip(range(len(overlapWordsStem)),overlapWordsStem):
        if overlapScoresStem[i][1] == -1:
            overlapScoresSplitStem[0].append(overlapScoresStem[i][0])
            overlapWordsSplitStem[0].append(word)
        elif overlapScoresStem[i][1] == 1:
            overlapScoresSplitStem[2].append(overlapScoresStem[i][0])
            overlapWordsSplitStem[2].append(word)
        else:
            overlapScoresSplitStem[1].append(overlapScoresStem[i][0])
            overlapWordsSplitStem[1].append(word)

    if not x_set.title == y_set.title:
        # let's save the buckets of words...
        # for the full-full I made two lists each
        # now, could make 9...
        # let's just do 3
        # they're stored above as scores with x,y
        # (note that for binning into the split, using the y score)
        def mash(idx,scr,binscr):
            overlapScoresSplit_combined = overlapScoresSplit[idx]+overlapScoresSplitStem[idx]
            overlapWordsSplit_combined = overlapWordsSplit[idx]+overlapWordsSplitStem[idx]
            indexer = sorted(range(len(overlapScoresSplit_combined)),key=lambda x: overlapScoresSplit_combined[x])
            overlapScoresSplit_combined_sorted = [overlapScoresSplit_combined[i] for i in indexer]
            overlapWordsSplit_combined_sorted = [overlapWordsSplit_combined[i] for i in indexer]
            # left_bin = [(overlapScoresSplit_combined_sorted[i],overlapWordsSplit_combined_sorted[i]) for i in range(len(overlapWordsSplit_combined_sorted)) if overlapScoresSplit_combined_sorted[i] == -1]
            # middle_bin = [(overlapScoresSplit_combined_sorted[i],overlapWordsSplit_combined_sorted[i]) for i in range(len(overlapWordsSplit_combined_sorted)) if overlapScoresSplit_combined_sorted[i] == 0]
            # right_bin = [(overlapScoresSplit_combined_sorted[i],overlapWordsSplit_combined_sorted[i]) for i in range(len(overlapWordsSplit_combined_sorted)) if overlapScoresSplit_combined_sorted[i] == 1]
            match_bin = [(overlapScoresSplit_combined_sorted[i],overlapWordsSplit_combined_sorted[i]) for i in range(len(overlapWordsSplit_combined_sorted)) if overlapScoresSplit_combined_sorted[i] == binscr]

            # don't reall need the csv....
            # f = open("txt-output/{0}-{2:+d}-{1}-all-bin.csv".format(y_set.title,x_set.title,scr),"w")
            # f.write('word,score1\n')
            # for score,word in match_bin:
            #     f.write('{0},{1:.2f}\n'.format(word,score))
            # f.close()

            write_table(match_bin,scr)

        def write_table(my_bin,my_score):
            f = open("{0}-{2:+d}-{1}-all-bin.tex".format(y_set.title,x_set.title,my_score),"w")
            f.write('\\begin{tabular}[t]{p{1.5cm}|p{1.0cm}|p{1.0cm}}\n')
            f.write('Word & $h_{{\\text{{ {0} }} }}$ & $h_{{\\text{{ {1} }} }}$ \\\\\n\\hline\n'.format(y_set.title,x_set.title))
            for score,word in my_bin:
                f.write('{0} & \\centering\\arraybackslash {1:+.0f} & \\centering\\arraybackslash {2:+.0f} \\\\\n'.format(word,score,my_score))                
            f.write('\\end{tabular}\n')
            f.close()

            tabletex_file_to_pdf("{0}-{2:+d}-{1}-all-bin.tex".format(y_set.title,x_set.title,my_score))

        # mash(idx,scr,binscr):
        mash(0,-1,+1)

        # mash(1,1,+1)

        mash(2,+1,-1)

        tex_files = ["{0}-{2:+d}-{1}-all-bin.tex".format(y_set.title,x_set.title,-1),"{0}-{2:+d}-{1}-all-bin.tex".format(y_set.title,x_set.title,+1)]
        pdf_files = ["{0}-{2:+d}-{1}-all-bin.pdf".format(y_set.title,x_set.title,-1),"{0}-{2:+d}-{1}-all-bin.pdf".format(y_set.title,x_set.title,+1)]        
        titles = ["\"\"","\"\"",]

        tabletile("1 2 .3 1 1 \"p{4cm}\" 10 \"\" scriptsize",tex_files,titles,"{}-horizontal".format(tile))
        tabletile("2 1 .3 1 1 \"p{4cm}\" 10 \"\" scriptsize",tex_files,titles,"{}-vertical".format(tile))
        pdftile("1 2 .3 1 1 l 10 \"\"",pdf_files,titles,"{}-horizontal-pdf".format(tile))
        pdftile("2 1 .3 1 1 l 10 \"\"",pdf_files,titles,"{}-vertical-pdf".format(tile))
        
    # now go plot
    nbins = [-1.5,-.5,.5,1.5]
    ax1 = axlist[0]
    ax1.hist([overlapScoresSplit[0],overlapScoresSplitStem[0]],bins=nbins,alpha=0.75,stacked=True)
    ax1.set_xlim([-1.5,1.5])
    ax1.set_xticks([])
    # ax1.set_yticks([20,40,60])
    ax1.set_yticks([])
    ax1.yaxis.set_ticks_position('right')
    ax2 = axlist[1]
    if len(overlapScoresSplit[1]) > 0:
        ax2.hist([overlapScoresSplit[1],overlapScoresSplitStem[1]],bins=nbins,alpha=0.75,stacked=True)
    ax2.set_xlim([-1.5,1.5])
    ax2.set_xticks([])
    # ax2.set_yticks([50,150,250])
    # ax2.set_yticks([20,40,60])
    ax2.set_yticks([])
    ax2.yaxis.set_ticks_position('right')
    ax3 = axlist[2]
    ax3.hist([overlapScoresSplit[2],overlapScoresSplitStem[2]],bins=nbins,alpha=0.75,stacked=True)
    ax3.set_xlim([-1.5,1.5])
    # ax3.xaxis.set_ticks_position('top')
    ax3.yaxis.set_ticks_position('right')
    ax3.set_xticks([])
    # ax3.set_yticks([20,40,60])
    # ax3.set_ylim([0,60])
    ax3.set_yticks([])
    
    axlims = (ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1])
    print(axlims)
    maxylim = np.max(axlims)
    ax1.set_ylim([0,maxylim])
    ax2.set_ylim([0,maxylim])
    ax3.set_ylim([0,maxylim])
    
    # x = np.linspace(min(x_setscores),max(x_setscores),num=100)
    # ax.plot(x,myoutput.beta[0]*x+myoutput.beta[1],'r',linewidth=0.75)
    
    # ax.legend(['RMA $\\beta$ ={0:.2f}'.format(myoutput.beta[0])],loc='best',fontsize=10)

    # ax1.set_xlabel('LabMT',fontsize=14)
    # ax1.set_ylabel('WK',fontsize=14)
    # ax.set_xlim([1,9])
    # ax.set_ylim([1,9])
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    return maxylim

def scatter_full_integer_3(x_set,y_set,axlist,orientation='vertical',tile="A"):
    # now go and make the triple histogram!
    overlapWordsFixed = []
    overlapScoresFixed = []
    # capture the num of stem matches
    overlapWordsStem = []
    overlapScoresStem = []
    print("x is a {0}".format(x_set.datastructure))
    print("y is a {0}".format(y_set.datastructure))    
    if x_set.datastructure == 'dict':
        # this means that x is a dict
        # and y is a trie
        # still using x to match all words in y
        for word,index in y_set.my_marisa[0].items():
            # exact fixed matches
            if word in x_set.data:
                score = x_set.data[word][1]
                overlapWordsFixed.append(word)
                overlapScoresFixed.append((score,index[1]))
        for word,index in y_set.my_marisa[1].items():
            # exact stem matches
            if word in x_set.data:
                score = x_set.data[word][1]
                overlapWordsFixed.append(word)
                overlapScoresFixed.append((score,index[1]))
    else:
        # now we know that y is a dict
        # and x is a trie
        # using x to match all words in y
        for word,index in y_set.data.items():
            # exact fixed matches
            # print(x_set.data)
            if word in x_set.my_marisa[0]:
                score = x_set.my_marisa[0][word][0][1]
                overlapWordsFixed.append(word)
                overlapScoresFixed.append((index[1],score))
            # this uses x_set.data's stems to search y_set.data's fixed words
            elif len(x_set.my_marisa[1].prefixes(word)) > 0:
                score = x_set.my_marisa[1].get(x_set.my_marisa[1].prefixes(word)[0])[0][1]
                overlapWordsStem.append(word)
                overlapScoresStem.append((index[1],score))

    print(overlapScoresFixed[:10])
    print(overlapWordsFixed[:10])
    print(overlapScoresStem[:10])
    print(overlapWordsStem[:10])
    overlapScoresSplitCombined = [[],[],[]]
    overlapWordsSplitCombined = [[],[],[]]
    overlapScoresSplitFixed = [[],[],[]]
    overlapWordsSplitFixed = [[],[],[]]
    for i,word in zip(range(len(overlapWordsFixed)),overlapWordsFixed):
        if overlapScoresFixed[i][1] == -1:
            overlapScoresSplitFixed[0].append(overlapScoresFixed[i][0])
            overlapWordsSplitFixed[0].append(word)
            overlapScoresSplitCombined[0].append(overlapScoresFixed[i][0])
            overlapWordsSplitCombined[0].append(word)            
        elif overlapScoresFixed[i][1] == 1:
            overlapScoresSplitFixed[2].append(overlapScoresFixed[i][0])
            overlapWordsSplitFixed[2].append(word)
            overlapScoresSplitCombined[2].append(overlapScoresFixed[i][0])
            overlapWordsSplitCombined[2].append(word)
        else:
            overlapScoresSplitFixed[1].append(overlapScoresFixed[i][0])
            overlapWordsSplitFixed[1].append(word)
            overlapScoresSplitCombined[1].append(overlapScoresFixed[i][0])
            overlapWordsSplitCombined[1].append(word)
    overlapScoresSplitStem = [[],[],[]]
    overlapWordsSplitStem = [[],[],[]]
    for i,word in zip(range(len(overlapWordsStem)),overlapWordsStem):
        if overlapScoresStem[i][1] == -1:
            overlapScoresSplitStem[0].append(overlapScoresStem[i][0])
            overlapWordsSplitStem[0].append(word)
            overlapScoresSplitCombined[0].append(overlapScoresStem[i][0])
            overlapWordsSplitCombined[0].append(word)            
        elif overlapScoresStem[i][1] == 1:
            overlapScoresSplitStem[2].append(overlapScoresStem[i][0])
            overlapWordsSplitStem[2].append(word)
            overlapScoresSplitCombined[2].append(overlapScoresStem[i][0])
            overlapWordsSplitCombined[2].append(word)            
        else:
            overlapScoresSplitStem[1].append(overlapScoresStem[i][0])
            overlapWordsSplitStem[1].append(word)
            overlapScoresSplitCombined[1].append(overlapScoresStem[i][0])
            overlapWordsSplitCombined[1].append(word)
            
    # overlapScoresSplitCombined = [[],[],[]]
    # overlapWordsSplitCombined = [[],[],[]]
    print(overlapScoresSplitCombined[0][:10])
    print(overlapScoresSplitCombined[1][:10])
    print(overlapScoresSplitCombined[2][:10])
    print(overlapWordsSplitCombined[0][:10])
    print(overlapWordsSplitCombined[1][:10])
    print(overlapWordsSplitCombined[2][:10])

    if not x_set.title == y_set.title:
        # let's save the buckets of words...
        # for the full-full I made two lists each
        # (note that for binning into the split, using the y score)
        indexer = sorted(range(len(overlapScoresSplitCombined[0])),key=lambda x: overlapScoresSplitCombined[0][x])
        overlapScoresSplit_combined_sorted = [overlapScoresSplitCombined[0][i] for i in indexer]
        overlapWordsSplit_combined_sorted = [overlapWordsSplitCombined[0][i] for i in indexer]
        f = open("txt-output/{0}--1-{1}-all-bin.csv".format(x_set.title,y_set.title),"w")
        f.write('word,score1\n')
        for score,word in zip(overlapScoresSplit_combined_sorted,overlapWordsSplit_combined_sorted):
            f.write('{0},{1:.2f}\n'.format(word,score))
        f.close()

        def write_table(a,b,my_score):
            f = open("{0}-{2:+d}-{1}-all-bin.tex".format(x_set.title,y_set.title,my_score),"w")
            f.write('\\begin{tabular}[t]{p{1.5cm}|p{1.0cm}|p{1.0cm}}\n')
            if x_set.score_range == 'integer':
                f.write('Word & $h_{{\\text{{ {1} }} }}$ & $h_{{\\text{{ {0} }} }}$ \\\\ \n'.format(x_set.title,y_set.title))
            else:
                f.write('Word & $h_{{\\text{{ {0} }} }}$ & $h_{{\\text{{ {1} }} }}$ \\\\ \n'.format(x_set.title,y_set.title))
            f.write('\\hline\n')
            for score,word in zip(a,b):
                if not my_score == 0:
                    # keep the sign there
                    f.write('{0} & \\centering\\arraybackslash {1:.2f} & \\centering\\arraybackslash {2:+.0f} \\\\\n'.format(word,score,my_score))
                else:
                    f.write('{0} & \\centering\\arraybackslash {1:.2f} & \\centering\\arraybackslash 0 \\\\\n'.format(word,score))
            f.write('\\end{tabular}\n')
            f.close()

            tabletex_file_to_pdf("{0}-{2:+d}-{1}-all-bin.tex".format(x_set.title,y_set.title,my_score))

        write_table(overlapScoresSplit_combined_sorted[-10:],overlapWordsSplit_combined_sorted[-10:],-1)
    
        indexer = sorted(range(len(overlapScoresSplitCombined[1])),key=lambda x: overlapScoresSplitCombined[1][x])
        overlapScoresSplit_combined_sorted = [overlapScoresSplitCombined[1][i] for i in indexer]
        overlapWordsSplit_combined_sorted = [overlapWordsSplitCombined[1][i] for i in indexer]    
        f = open("txt-output/{0}-0-{1}-all-bin.csv".format(x_set.title,y_set.title),"w")
        f.write('word,score1\n')
        for score,word in zip(overlapScoresSplit_combined_sorted,overlapWordsSplit_combined_sorted):
            f.write('{0},{1:.2f}\n'.format(word,score))
        f.close()

        write_table(overlapScoresSplit_combined_sorted[:5]+overlapScoresSplit_combined_sorted[-5:],overlapWordsSplit_combined_sorted[:5]+overlapWordsSplit_combined_sorted[-5:],+0)
    
        indexer = sorted(range(len(overlapScoresSplitCombined[2])),key=lambda x: overlapScoresSplitCombined[2][x])
        overlapScoresSplit_combined_sorted = [overlapScoresSplitCombined[2][i] for i in indexer]
        overlapWordsSplit_combined_sorted = [overlapWordsSplitCombined[2][i] for i in indexer]    
        f = open("txt-output/{0}-+1-{1}-all-bin.csv".format(x_set.title,y_set.title),"w")
        f.write('word,score1\n')
        for score,word in zip():
            f.write('{0},{1:.2f}\n'.format(word,score))
        f.close()

        write_table(overlapScoresSplit_combined_sorted[:10],overlapWordsSplit_combined_sorted[:10],1)

        tex_files = ["{0}-{2:+d}-{1}-all-bin.tex".format(x_set.title,y_set.title,-1),
                     "{0}-{2:+d}-{1}-all-bin.tex".format(x_set.title,y_set.title,0),
                     "{0}-{2:+d}-{1}-all-bin.tex".format(x_set.title,y_set.title,+1),]
        pdf_files = ["{0}-{2:+d}-{1}-all-bin.pdf".format(x_set.title,y_set.title,-1),
                     "{0}-{2:+d}-{1}-all-bin.pdf".format(x_set.title,y_set.title,0),
                     "{0}-{2:+d}-{1}-all-bin.pdf".format(x_set.title,y_set.title,+1),]
        titles = ["\"\"","\"\"","\"\"",]

        tabletile("1 3 .3 1 1 \"p{4cm}\" 10 \"\" scriptsize",tex_files,titles,"{}-horizontal".format(tile))
        tabletile("3 1 .3 1 1 \"p{4cm}\" 10 \"\" scriptsize",tex_files,titles,"{}-vertical".format(tile))
        pdftile("1 3 .3 1 1 l 10 \"\"",pdf_files,titles,"{}-horizontal-pdf".format(tile))
        pdftile("3 1 .3 1 1 l 10 \"\"",pdf_files,titles,"{}-vertical-pdf".format(tile))

    nbins = np.linspace(1,9,8./.15)
    
    ax1 = axlist[0]
    ax1.hist([overlapScoresSplitFixed[0],overlapScoresSplitStem[0]],bins=nbins,alpha=0.75,orientation=orientation,stacked=True)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2 = axlist[1]
    if len(overlapScoresSplitCombined[1]) > 0:
        ax2.hist([overlapScoresSplitFixed[1],overlapScoresSplitStem[1]],bins=nbins,alpha=0.75,orientation=orientation,stacked=True)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax3 = axlist[2]
    ax3.hist([overlapScoresSplitFixed[2],overlapScoresSplitStem[2]],bins=nbins,alpha=0.75,orientation=orientation,stacked=True)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    if orientation == 'vertical':
        ax1.set_xlim([1,9])
        ax2.set_xlim([1,9])
        ax3.set_xlim([1,9])
        axlims = (ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1])
        # print(axlims)
        maxylim = np.max(axlims)
        ax1.set_ylim([0,maxylim])
        ax2.set_ylim([0,maxylim])
        ax3.set_ylim([0,maxylim])
    else:
        ax1.set_ylim([1,9])
        ax2.set_ylim([1,9])
        ax3.set_ylim([1,9])
        axlims = (ax1.get_xlim()[1],ax2.get_xlim()[1],ax3.get_xlim()[1])
        # print(axlims)
        maxylim = np.max(axlims)
        ax1.set_xlim([0,maxylim])
        ax2.set_xlim([0,maxylim])
        ax3.set_xlim([0,maxylim])
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        ax3.invert_xaxis()        
    
    return maxylim

def scatterall3(upper=True,inches=12,labelsize=12,insetsize=10,upperHist=False,letterboxsize=12):
    fig = plt.figure(figsize=(inches,inches))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # dataList = [LabMT_dict,ANEW_dict,WK_dict,MPQA_trie,LIWC_trie,Liu_trie]
    dataList = [my_LabMT,my_ANEW,my_WK,my_MPQA,my_LIWC,my_Liu]
    for a in dataList:
        print(a.datastructure)

    # dataList = [LabMT_dict,WK_dict,ANEW_dict]

    plotsize = len(dataList)-1
    # whole figure label padding
    xpad = .085
    ypad = .085
    xpadr = .015
    ypadr = .015
    # remaining width
    xrem = 1.-xpad-xpadr
    yrem = 1.-ypad-ypadr
    # divide it up
    if upper:
        xwidth = xrem/(plotsize+1)
        ywidth = xrem/(plotsize+1)
    else:
        xwidth = xrem/(plotsize)
        ywidth = xrem/(plotsize)
    # go down
    if upper:
        istart = 0
    else:
        istart = 1
    # i marches down
    for i in range(istart,plotsize+1):
        if upper:
            jrange=plotsize+1
        else:
            jrange=i
        # j marches across
        for j in range(0,jrange):
            print('-'*80)
            print('-'*80)
            print('i={0},j={1}'.format(i,j))
            print('set 1 is {0}, set 2 is {1}'.format(dataList[j].title,dataList[i].title))
            print('set 1 is type {0}, set 2 is type {1}'.format(dataList[j].score_range,dataList[i].score_range))
            rect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth,xwidth,ywidth]
            # print(rect)
            if dataList[j].score_range == 'full' and dataList[i].score_range == 'full':
                ax = fig.add_axes(rect)
                scatter_full_full(dataList[j],dataList[i],ax,insetsize=insetsize,tile=letters[(j+0)+(i)*6])
                if i == plotsize and upper:
                    ax.set_xlim([1,9])
                    # ax.set_xticks(range(1,10))
                    ax.set_xticks(range(2,9))
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                elif i == plotsize:
                    ax.set_xlim([1,9])
                    # ax.set_xticks(range(1,10))
                    ax.set_xticks(range(2,9))
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                else:
                    ax.set_xticks([])
            elif dataList[j].score_range == 'full' and dataList[i].score_range == 'integer':
                # take the rectangle, divide into three vertically
                ax = fig.add_axes(rect)
                if i == plotsize and upper:
                    ax.set_xlim([1,9])
                    ax.set_xticks(range(2,9))
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                elif i == plotsize:
                    ax.set_xlim([1,9])
                    if not j == 1:
                        ax.set_xticks(range(1,10))
                    else:
                        ax.set_xticks(range(2,9))
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                negrect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth,xwidth,ywidth/4.]
                neurect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth+ywidth/3.,xwidth,ywidth/4.]
                posrect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth+2*ywidth/3.,xwidth,ywidth/4.]
                axlist = (fig.add_axes(negrect,frame_on=False),
                          fig.add_axes(neurect,frame_on=False),
                          fig.add_axes(posrect,frame_on=False))
                print('using scatter_full_integer_3')
                # print(rectlist)
                maxylim = scatter_full_integer_3(dataList[j],dataList[i],axlist,orientation='vertical',tile=letters[(j+0)+(i)*6])
                ax.text(0.05, 0.95, '$N_{{\max}}$ = {0:.0f}'.format(maxylim),
                        transform=ax.transAxes, fontsize=letterboxsize,
                        verticalalignment='top', bbox=None)
            elif dataList[j].score_range == 'integer' and dataList[i].score_range == 'full':
                # take the rectangle, divide into three vertically
                ax = fig.add_axes(rect)
                if i == plotsize and upper:
                    ax.set_xlim([1,9])
                    ax.set_xticks(range(2,9))
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                elif i == plotsize:
                    ax.set_xlim([1,9])
                    ax.set_xticks(range(2,9))
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                negrect = [xpad+j*xwidth+(xwidth/3.-xwidth/4.),ypad+(plotsize-i)*ywidth,xwidth/4.,ywidth]
                neurect = [xpad+j*xwidth+xwidth/3.+(xwidth/3.-xwidth/4.),ypad+(plotsize-i)*ywidth,xwidth/4.,ywidth]
                posrect = [xpad+j*xwidth+2*xwidth/3.+(xwidth/3.-xwidth/4.),ypad+(plotsize-i)*ywidth,xwidth/4.,ywidth]
                axlist = (fig.add_axes(negrect,frame_on=False),
                          fig.add_axes(neurect,frame_on=False),
                          fig.add_axes(posrect,frame_on=False))
                axlist[0].invert_yaxis()
                print('using scatter_full_integer_3')
                # print(rectlist)
                maxylim = scatter_full_integer_3(dataList[j],dataList[i],axlist,orientation='horizontal',tile=letters[(j+0)+(i)*6])
                ax.text(0.05, 0.95, '$N_{{\max}}$ = {0:.0f}'.format(maxylim), transform=ax.transAxes,
                        fontsize=letterboxsize, verticalalignment='top', horizontalalignment='left',
                        bbox=None, rotation=90)
                # note that rotation is applied after the position is set
            elif dataList[j].score_range == 'integer' and dataList[i].score_range == 'integer':
                # take the rectangle, divide into three vertically
                ax = fig.add_axes(rect)
                if i == plotsize and upper:
                    # ax.set_xlim([1,9])
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                    ax.set_xlim([-1.5,1.5])
                    ax.set_xticks([-1,0,1])
                elif i == plotsize:
                    # ax.set_xlim([1,9])
                    ax.set_xlabel(dataList[j].title,fontsize=labelsize)
                    ax.set_xlim([-1.5,1.5])
                    ax.set_xticks([-1,0,1])
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                negrect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth,xwidth,ywidth/4.]
                neurect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth+ywidth/3.,xwidth,ywidth/4.]
                posrect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth+2*ywidth/3.,xwidth,ywidth/4.]
                axlist = (fig.add_axes(negrect,frame_on=False),fig.add_axes(neurect,frame_on=False),fig.add_axes(posrect,frame_on=False))
                print('using scatter_integer_integer_3')
                # print(rectlist)
                maxylim = scatter_integer_integer_3(dataList[j],dataList[i],axlist,tile=letters[(j+0)+(i)*6])
                ax.text(0.05, 0.95, '$N_{{\max}}$ = {0:.0f}'.format(maxylim), transform=ax.transAxes, fontsize=letterboxsize,
                    verticalalignment='top', bbox=None)
            if j == 0:
                ax.set_ylabel(dataList[i].title,fontsize=labelsize)
                if dataList[i].score_range == 'full':
                    ax.set_ylim([1,9])
                    # ax.set_yticks(range(1,10))
                    ax.set_yticks(range(2,9))
                if dataList[i].score_range == 'integer':
                    ax.set_ylim([-1.5,1.5])
                    ax.set_yticks([-1,0,1])
            # put a label onto each of the axis
            # this is the full rect:
            # rect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth,xwidth,ywidth]
            # this is a much smaller rect, in the bottom right-ish
            label_rect_width = xwidth/6
            # print("-"*200)
            # print(xwidth) # .15
            rect = [xpad+(j+1)*xwidth-.01,ypad+(plotsize-i)*ywidth+.01,label_rect_width,label_rect_width]
            props = dict(boxstyle='square', facecolor='white', alpha=1.0)
            fig.text(rect[0], rect[1], letters[(j+0)+(i)*6],
                     fontsize=letterboxsize,
                     verticalalignment='bottom',
                     horizontalalignment='right',
                     bbox=props)
    if upperHist:
        i = 1
        j = 1 
        rect = [xpad+j*xwidth,ypad+(plotsize-i)*ywidth,xwidth,ywidth]
        print('top corner')
        ax = fig.add_axes(rect)
        histogram_WK_LabMT(dataList[1].data,dataList[0].data,ax,insetsize=insetsize)

    # print("not saving figure")
    mysavefig('scatter-{0}.pdf'.format('-'.join([a.title for a in dataList])))

letters = [x.upper() for x in ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","aa","bb","cc","dd","ee","ff","gg","hh","ii","jj","kk","ll","mm","nn","oo","pp","qq","rr","ss","tt","uu","vv","xx","yy","zz"]]

def scatterReplyHorizontal(inches=12,labelsize=12,insetsize=6):
    fig = plt.figure(figsize=(inches,inches/3))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    dataList = [LabMT_dict,WK_dict,ANEW_dict]

    plotsize = len(dataList)
    # whole figure label padding
    xpad = .01
    ypad = .15

    xpadr = .01
    xpadi = .05

    ypadr = .03
    # remaining width
    xrem = 1.-xpad-xpadr
    yrem = 1.-ypad-ypadr
    # divide it up
    xwidth = xrem/(plotsize)
    ywidth = yrem
 
    i = 0
    j = 0
    print('i={0},j={1}'.format(i,j))
    print('set 1 is {0}, set 2 is {1}'.format(dataList[j].title,dataList[i].title))
    rect = [xpad+xpadi+j*(xwidth),ypad,xwidth-xpadi,ywidth]
    # print(rect)
    ax = fig.add_axes(rect)
    scatter_full_full(LabMT_dict.data,WK_dict.data,ax,insetsize=insetsize)
    ax.set_xlim([1,9])
    ax.set_xticks(range(1,10))
    ax.set_ylim([1,9])
    ax.set_yticks(range(1,10))
    ax.set_xlabel('WK',fontsize=labelsize)
    ax.set_ylabel('LabMT',fontsize=labelsize)

    i = 0
    j = 1
    print('i={0},j={1}'.format(i,j))
    print('set 1 is {0}, set 2 is {1}'.format(dataList[j].title,dataList[i].title))
    rect = [xpad+xpadi+j*(xwidth),ypad,xwidth-xpadi,ywidth]
    # print(rect)
    ax = fig.add_axes(rect)
    scatter_full_full(LabMT_dict.data,ANEW_dict.data,ax,insetsize=insetsize)
    ax.set_xlim([1,9])
    ax.set_xticks(range(1,10))
    ax.set_ylim([1,9])
    ax.set_yticks(range(1,10))
    ax.set_xlabel('ANEW',fontsize=labelsize)
    ax.set_ylabel('LabMT',fontsize=labelsize)

    i = 0
    j = 2
    print('i={0},j={1}'.format(i,j))
    print('set 1 is {0}, set 2 is {1}'.format(dataList[j].title,dataList[i].title))
    rect = [xpad+xpadi+j*(xwidth),ypad,xwidth-xpadi,ywidth]
    # print(rect)
    ax = fig.add_axes(rect)
    scatter_full_full(ANEW_dict.data,WK_dict.data,ax,insetsize=insetsize)
    ax.set_xlim([1,9])
    ax.set_xticks(range(1,10))
    ax.set_ylim([1,9])
    ax.set_yticks(range(1,10))
    ax.set_xlabel('WK',fontsize=labelsize)
    ax.set_ylabel('ANEW',fontsize=labelsize)

    mysavefig('scatter-{0}.pdf'.format('-'.join([a.title for a in dataList])))

if __name__ == '__main__':
    # For the comparison of two dictionaries, we plot words that are matched by the independent variable (x) in the dependent variable (y).
    # scatterall3()
    scatterall3(inches=8.5,labelsize=12,insetsize=8,letterboxsize=10)

    # yes, makes the horizontal figure for the reply
    # scatterReplyHorizontal(inches=10.,labelsize=15.,insetsize=10.)

    # dump out the the comparison word lists
    # make_shift_plots_all()



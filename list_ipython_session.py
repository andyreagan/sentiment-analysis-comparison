# coding: utf-8
from scatterplot import *
fig = plt.figure()
overLapWordsSplit = scatter1(LabMT.data,LIWCtrie.data,fig)
print(len(overLapWordsSplit[0]))
print(len(overLapWordsSplit[1]))
print(len(overLapWordsSplit[2]))
print(overLapWordsSplit[1][:10])
f = open("data/LIWC/LIWC2007_English100131_words.dic","r")
LIWC = dict()
for line in f:
    l = line.rstrip().split("\t")
    word = l[0]
    if "1" in l:
        LIWC[word] = l[1:]
print(len(LIWC))
print(len(overLapWordsSplit[1]))
print("quickly" in overLapWordsSplit[1])
print(overLapWordsSplit[1][:10])
f.close()
f = open('LIWC-function-matches-LabMT','w')
for word in overLapWordsSplit[1]:
    f.write('{0}\n'.format(word))
for word in overLapWordsSplit[1]:
    f.write('{0}\n'.format(word))
    
f.close()
overLapScoresSplit = [LabMT.data[word] for word in overLapWordsSplit[1]]
overLapScoresSplit
overLapScoresSplit = [LabMT.data[word] for word in overLapWordsSplit[1]]
f = open('LIWC-function-matches-LabMT-whapps.txt','w')
for word,scorelist in zip(overLapWordsSplit[1],overLapScoresSplit):
    f.write('{0},{1},{2}\n'.format(word,scorelist[0],scorelist[2]))
    
f.close()
stemindexer = sorted(range(len(overLapScoresSplit)), key=lambda k: overLapScoresSplit[k], reverse=True)
wordsSorted = [overLapScoresSplit[i] for i in stemindexer]
scoresSorted = [overLapScoresSplit[i] for i in stemindexer]
wordsSorted = [overLapWordsSplit[1][i] for i in stemindexer]
print(scoresSorted[:10])
print(wordsSorted[:10])
allSorted = [(word,score[0],score[2]) for word,score in zip(wordsSorted,scoresSorted)]
print(allSorted[:10])
f = open('LIWC-function-matches-LabMT-whapps-sorted.txt','w')
for x in allSorted:
    f.write('{0},{1:.2f},{2:.2f}\n'.format(x[0],x[1],x[2]))
    
f.close()
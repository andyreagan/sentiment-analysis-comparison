# accept a timestamp
# read the current date
# pump out a qsub script named by the timestamp, for the current date

import datetime
import sys
import subprocess
import time
from os.path import isdir,isfile
from os import mkdir

max_jobs = 150

jobs = int(subprocess.check_output("showq | grep areagan | wc -l",shell=True))
print("there are {0} jobs via showq".format(jobs))

submitted_count = 0

while jobs+submitted_count < max_jobs:
    # ctime = subprocess.check_output("date +%S.%M.%H.%d.%m.%y",shell=True).rstrip()
    # job submission is synchronous.... so don't worry about writing out different files
    # the only reason to write out the shell script in the first place is to make sure that env variables make it to the job
    ctime = "tmp"
    
    f = open("currdate.txt","r")
    currdate = datetime.datetime.strptime(f.read(),"%Y-%m-%d")
    f.close()

    # add a day and save it
    nextday = currdate+datetime.timedelta(days=1)
    f = open("currdate.txt","w")
    f.write(nextday.strftime("%Y-%m-%d"))
    f.close()
    

    script = '''export DATE={0}
qsub -qshortq -V run.qsub
\\rm {1}.sh

'''.format(currdate.strftime('%Y-%m-%d'),ctime)

    
            
    # print('writing {}.sh'.format(ctime))
    f = open('{}.sh'.format(ctime),'w')
    f.write(script)
    f.close()
    subprocess.call(". {}.sh".format(ctime),shell=True)
    
    time.sleep(.25)
    submitted_count += 1











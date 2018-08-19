import Pyxis
import ms
import numpy
import os
import glob
import time
from pyrap.tables import table


def gi(message):
	print '\033[92m'+message+'\033[0m'


def bi(message):
	print '\033[94m\033[1m'+message+'\033[0m'


def ri(message):
    print '\033[91m'+message+'\033[0m'


def predict(msname,imgbase):
        syscall = 'wsclean -predict -channelsout 12 -size 8192 8192 '
        syscall+= '-scale 1.5asec -name '+imgbase+' -mem 80 '
        syscall+= '-predict-channels 12 '+msname+' >> wspredict.log'
        os.system(syscall)


def add_scratch_col(myms,col_name):
	tt = table(myms,readonly=False,ack=False)
	desc = tt.getcoldesc('DATA')
	desc['name'] = col_name
	desc['comment'] = desc['comment'].replace(' ','_')
	tt.addcols(desc)
	tt.done()
	return desc

def add_scratch_col(myms,col_name):
	tt = table(myms,readonly=False,ack=False)
	colnames = tt.colnames()
	if col_name in colnames:
		bi(col_name+' already exists, will not be created')
	else:
		desc = tt.getcoldesc('DATA')
		desc['name'] = col_name
		desc['comment'] = desc['comment'].replace(' ','_')
		tt.addcols(desc)
	tt.done()
	return desc

myms = 'laduma_01_2048_wtspec_J033230-280757.ms'
prefix = 'img_laduma_01_2048_wtspec_J033230-280757.ms_corr_pcal'
dryrun = False

models = sorted(glob.glob(prefix+'*weighted.fits'))
tempname = 'tempmodel'

t_total = 0.0

chans = []
dirs = []

for mod in models:
	chan = mod.split('-model')[0].split('-')[-1]
	chans.append(chan)
	mydir = mod.split('_dir')[-1].split('_')[0]
	dirs.append('dir'+mydir)

chans = numpy.unique(chans)
dirs = numpy.unique(dirs)

for mydir in dirs[::-1]: # want dir0 in MODEL_DATA
	t0 = time.time()
	mycol = mydir.upper()
	bi('Direction '+mydir)
	for chan in chans:
		modimage = glob.glob(prefix+'*'+chan+'*'+mydir+'*weighted.fits')[0]
		gi('    Chan '+chan+' model image: '+modimage)
		tmpimg = tempname+'-'+chan+'-model.fits'
		if not dryrun:
			os.system('cp '+modimage+' '+tmpimg)
	gi('    Running wsclean prediction')
	if not dryrun:
		predict(myms,tempname)
	if mydir != dirs[0]:
		gi('    Adding scratch column '+mycol)
		if not dryrun:
			add_scratch_col(myms,mycol)
		gi('    Copying MODEL_DATA to '+mycol)
		if not dryrun:
			v.MS = myms
			ms.copycol(fromcol='MODEL_DATA',tocol=mycol)
	else:
		gi('    Direction 0 prediction remains in MODEL_DATA')
	t1 = time.time()
	tt = t1 - t0
	bi('Time for this direction: '+str(round(tt,2))+' sec')
	t_total += tt
bi('Done')
bi('Total time elapsed: '+str(round(t_total,2))+' sec')

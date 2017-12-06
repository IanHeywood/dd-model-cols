import matplotlib
matplotlib.use('Agg')
import glob
import numpy
import pylab
import os
import time
import aplpy
from astropy.io import fits
from random import randint

#=============================================================


c = 299792458.0


def gi(message):
	print '\033[92m'+message+'\033[0m'


def bi(message):
	print '\033[94m\033[1m'+message+'\033[0m'


def ri(message):
    print '\033[91m'+message+'\033[0m'


def spacer():
	gi('-------------------------------------------')
	bi('-------------------------------------------')


def getInfo(fitsfile):
	input_hdu = fits.open(fitsfile)[0]
	hdr = input_hdu.header
	nx = hdr.get('NAXIS1')
	ny = hdr.get('NAXIS2')
	dx = abs(hdr.get('CDELT1'))
	dy = abs(hdr.get('CDELT2'))
	if dx != dy:
		ri("Pixels aren't square, which might ruin everything")
	return nx,ny,dx


def getfreq(fitsfile):
	input_hdu = fits.open(fitsfile)[0]
	hdr = input_hdu.header
	freq = hdr.get('CRVAL3')
	return freq


def ang2pix(ang,pixscale):
	return ang/pixscale


def getImage(fitsfile):
	# Return the image data from fitsfile as a numpy array
	input_hdu = fits.open(fitsfile)[0]
	if len(input_hdu.data.shape) == 2:
		image = numpy.array(input_hdu.data[:,:])
	elif len(input_hdu.data.shape) == 3:
		image = numpy.array(input_hdu.data[0,:,:])
	else:
		image = numpy.array(input_hdu.data[0,0,:,:])
	return image


def flushFits(newimage,fitsfile):
	# Write numpy array newimage to fitsfile
	# Dimensions must match (obv)
	f = fits.open(fitsfile,mode='update')
	input_hdu = f[0]
	if len(input_hdu.data.shape) == 2:
		input_hdu.data[:,:] = newimage
	elif len(input_hdu.data.shape) == 3:
		input_hdu.data[0,:,:] = newimage
	else:
		input_hdu.data[0,0,:,:] = newimage
	f.flush()


def weight2fits(myfits,weight_img,dir):
	opfits = myfits.replace('.fits','_dir'+str(dir)+'_weight.fits')
	os.system('cp '+myfits+' '+opfits)
	flushFits(weight_img,opfits)
	return opfits


def molly(x):
	return numpy.exp(2.0 + (-2.0/(1.0-((x-1.0)**2))))


def inner(nx,ny,r0,r1):
	im = numpy.ones((ny,ny))
	for i in range(-nx/2,nx/2):
		for j in range(-nx/2,nx/2):
			rr = ((i**2.0)+(j**2.0))**0.5
			if rr < r0:
				val = 0.0
			elif rr > r1:
				val = 1.0
			else:
				rr = (rr-r0)/(r1-r0)
				if rr == 0.0:
					rr = 1e-15
				val = molly(rr)
			im[i-nx/2,j-nx/2] = 1.0 - val
	return im


def sector_mask(shape,centre,radius,angle_range):
	# https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
	"""
	Return a boolean mask for a circular sector. The start/stop angles in  
	`angle_range` should be given in clockwise order.
	"""
	x,y = numpy.ogrid[:shape[0],:shape[1]]
	cx,cy = centre
	tmin,tmax = numpy.deg2rad(angle_range)
	# ensure stop angle > start angle
	if tmax < tmin:
		tmax += 2*numpy.pi
	# convert cartesian --> polar coordinates
	r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
	theta = numpy.arctan2(x-cx,y-cy) - tmin
	# wrap angles between 0 and 2*pi
	theta %= (2*numpy.pi)
	# circular mask
	circmask = r2 <= radius*radius
	# angular mask
	anglemask = theta < (tmax-tmin)
	return circmask*anglemask


def make_weights(myfits,r0,r1,n_spokes,offset):

	weight_list = []

	nx,ny,dx = getInfo(myfits)
	inp_img = getImage(myfits)

	delta = 360.0/float(n_spokes)

	r0 = ang2pix(r0,dx)
	r1 = ang2pix(r1,dx)

	# Direction zero weights
	dir0 = inner(nx,ny,r0,r1)
	wtfits = weight2fits(myfits,dir0,0)
	#spacer()
	bi('     Direction 0 weight image: '+wtfits)
	weight_list.append(wtfits)
	# Rest of the image 
	rest = 1.0-dir0

	for i in range(0,n_spokes):
		theta0 = offset+(i*delta)
		theta1 = theta0+delta
		mask = sector_mask(rest.shape,(ny/2.0,nx/2.0),nx,(theta0,theta1))
		# Direction (i+1) mask
		sector = rest*mask
		wtfits = weight2fits(myfits,sector,i+1)
		bi('     Direction '+str(i+1)+' weight image: '+wtfits)
		weight_list.append(wtfits)

	return weight_list


def apply_weight(myfits,wtfits):
	opfits = wtfits.replace('weight','weighted')
	os.system('cp '+myfits+' '+opfits)
	inp_img = getImage(myfits)
	wt_img = getImage(wtfits)
	#spacer()
	bi('     Applying: '+wtfits)
	bi('     To:       '+myfits)
	op_img = inp_img * wt_img
	flushFits(op_img,opfits)
	totalflux = numpy.sum(op_img)
	return (opfits,totalflux)


def plot_areas(myfits,weight_list):
	pngname = 'plot_'+myfits+'_weight-areas.png'
	fig = pylab.figure(figsize=(32,32))
	f1 = aplpy.FITSFigure(myfits,slices=[0,0],figure=fig)#,subplot = [0.02,0.02,0.23,0.23])
	f1.show_colorscale(interpolation='none',cmap='Greys',stretch='linear')
	for fitsfile in weight_list:
		mycol = '#%06X' % randint(0, 0xFFFFFF)
		f1.show_contour(fitsfile,slices=[0,0],colors=[mycol],linewidths=[8.0],smoothing=3,levels=[0.5,1.0],alpha=0.7)
	f1.axis_labels.hide()
	f1.tick_labels.hide()
	f1.ticks.hide()
	f1.set_frame_color('white')
	fig.savefig(pngname,bbox_inches='tight')
	f1.close()
	return pngname


def first_null(f,D):
	null_rad = 0.61*(c/f)/D
	return 180.0*null_rad/numpy.pi


models = sorted(glob.glob('img_laduma_01_2048_wtspec_J033230-280757.ms_corr_pcal*00*-model.fits'))
n_spokes = 3
offset = 26.0
D = 13.5

t_total = 0.0

for myfits in models:
	t0 = time.time()
	freq = getfreq(myfits)
	null = first_null(freq,D)
	r0 = 0.9*null
	r1 = 1.1*null
	gi(myfits)
	gi('     freq = '+str(freq/1e9)+' GHz :: r0 = '+str(r0)+' deg :: f1 = '+str(r1)+' deg')
	weight_list = make_weights(myfits,r0,r1,n_spokes,offset)
	for wtfits in weight_list:
		result = apply_weight(myfits,wtfits)
		gi('     Sum of flux in '+result[0]+' is '+str(result[1])+' Jy')
	imgfits = myfits.replace('model','image')
	png = plot_areas(imgfits,weight_list)
	gi('Rendered: '+png)
	t1 = time.time()
	elapsed = round(t1-t0,2)
	gi('Elapsed time: '+str(elapsed)+' seconds')
	t_total+=elapsed
	spacer()

gi('Finished in '+str(round(t_total,2))+' seconds')

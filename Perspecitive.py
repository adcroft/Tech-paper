#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy
#%matplotlib inline
from Schematic_bergs import Schematic

class trans:
	def __init__(self, img, horizon_left_j, horizon_right_j, j_of_0, j_of_L, distance_to_image, y_L):
		self.ni = img.shape[1]
		self.nj = img.shape[0]
		self.horizon_left_j = horizon_left_j
		self.horizon_right_j = horizon_right_j
		self.horizon_center_j = 0.5*( horizon_left_j + horizon_right_j )
		self.j_of_0 = j_of_0
		self.j_of_L = j_of_L
		self.distance_to_image = distance_to_image # "d"
		self.y_L = y_L
		# Temporary values
		self.y_of_0 = 0.
		self.elevation = 1.
		# Derived quantities
		self.ic = 0.5*(self.ni-1)
		self.image_angle = math.atan( -self.w_j(self.horizon_center_j) / self.distance_to_image )
		print('Angle of image (alpha) = %f degree'%(self.image_angle * 180. / math.pi) )
		phi_L = math.atan( self.w_j(self.j_of_L) / self.distance_to_image )
		phi_0 = math.atan( self.w_j(self.j_of_0) / self.distance_to_image )
		self.elevation = self.y_L / ( math.tan( math.pi/2 - self.image_angle - phi_L )
					     - math.tan( math.pi/2 - self.image_angle - phi_0 ) )
		self.y_of_0 = self.y_j(self.j_of_0)
		self.dump()
	def dump(self):
        	for v in self.__dict__: print(v,'=',self.__dict__[v])
	def w_j(self, j):
        	"""Image coordinate for image index j"""
	        return j - 0.5*self.nj
	def j_w(self, w):
        	"""Image index for image coordinate w"""
	        return w + 0.5*self.nj
	def y_j(self, j):
       		"""Distance y for image index j"""
	        return self.y_w( self.w_j(j) )
    	def y_w(self, w):
        	"""Distance y for image coordinate w"""
	        phi = numpy.arctan( w / self.distance_to_image )
	        return self.elevation * numpy.tan( math.pi/2 - self.image_angle - phi ) - self.y_of_0
	def j_y(self, y):
        	"""Image index j for distance y"""
	        return self.j_w( self.w_y(y) )
	def w_y(self, y):
        	"""Image coordinate for distance y"""
	        theta = numpy.arctan( ( y + self.y_of_0 ) / self.elevation )
	        phi = math.pi/2 - self.image_angle - theta
	        return self.distance_to_image * numpy.tan( phi )
	def ij_xy(self, x, y, stretch=2):
        	"""Image index i,j for position x,y"""
	        w = self.w_y(y)
        	scale = stretch*numpy.sqrt( ( self.distance_to_image**2 + w**2 ) / ( self.elevation**2 + ( y + self.y_of_0 )**2 ) )
	        return self.ic + scale*x, self.j_w( w )
	def draw_equidistant(self, y=None, color_style='k:'):
        	"""Draws a horizon line or lateral at virtual x. y=None is at infinity."""
	        if y is None: j = self.horizon_center_j
        	else: j = self.j_y( y )
	        plt.plot([0,self.ni-1],[j,j],color_style)
	def hexagon_ij(self, cx, cy, r, rot=math.pi/6):
        	return self.polygon_ij(cx, cy, r, 6, rot=rot)
	def polygon_ij(self, cx, cy, r, nsegs, rot=0):
        	a = numpy.linspace(-math.pi+rot,math.pi+rot,nsegs+1)
	        x,y = cx + r*numpy.cos(a), cy + r*numpy.sin(a)
        	return self.ij_xy(x,y)
	def alpha(self,y):
        	yp = (self.y_L - y) / self.y_L
	        return 0.02 + 0.35 * ( yp )**2
	def xyz2ij(x,y,z):
        	"""Returns image coordinate (j,i) of a 3D position (x,y,z)"""
	        j = hor_c


def create_hexagon(T, grey, x, y, r, sample_radius=0.25, grey_threshold=0.66):
	"""Returns coords of hexagon if grey level is above threshold"""
	i,j = T.hexagon_ij(x,y,r*sample_radius)
	if i.min()>=0 and i.max()<T.ni and j.min()>=0 and j.max()<T.nj:
		
        	if grey[j.astype(int),i.astype(int)].mean()>grey_threshold:
            		ic,jc = T.ij_xy(x,y); 
            		return x,y,ic,jc
	return None,None,None,None

##########################################################  Main Program   #########################################################################
####################################################################################################################################################

def Perspective():
	#Reading in the image
	filename='Figures/P11B.png'
	#im = mpimg.imread('P11B.png')
	im = mpimg.imread(filename)
	
	# Grey level in image
	grey = numpy.sqrt((im[:,:,0]**2+im[:,:,1]**2+im[:,:,2]**2)/3)
	
	plt.figure(figsize=(16,8))
	plt.imshow(im);
	L,Lextra = 5.,5.
	T = trans(im,147,145,580,232,1,L)
	# Draws the horizon
	#T.draw_equidistant()
	# Draw an ocean grid
	for y in numpy.arange(-1.,L+Lextra+1,1.): T.draw_equidistant(y=y, color_style='c:')
	for x in numpy.arange(-L,L+1,1.):
		i,j = T.ij_xy([x,x],[-1,L+Lextra]); plt.plot(i,j,'c:')
	# Regular mesh of hexagon centers
	dd = 0.25; dt = math.sqrt(3./4.)
	x,y = numpy.arange(-L-.05,L/2.,dd), numpy.arange(-dd,L,dd*dt)
	x,y = numpy.meshgrid(x,y)
	x[::2,:] = x[::2,:] + 0.5*dd

	# Create hexagons
	hx,hy,hi,hj = numpy.zeros(x.shape),numpy.zeros(x.shape),numpy.zeros(x.shape),numpy.zeros(x.shape)
	for j in range(x.shape[0]):
		for i in range(x.shape[1]):
			hx[j,i],hy[j,i],hi[j,i],hj[j,i] = create_hexagon(T,grey,x[j,i],y[j,i],0.5*dd/dt)

	# Draw hexagons
	for j in range(x.shape[0]):
		for i in range(x.shape[1]):
			if hx[j,i] is not None:
		    		mi,mj = T.hexagon_ij(hx[j,i],hy[j,i],0.5*dd/dt)
				plt.plot(mi,mj,'r',alpha=T.alpha(hy[j,i]))
				plt.plot(hi[j,i],hj[j,i],'r.',alpha=T.alpha(hy[j,i]))

	# Draw bonds
	for j in range(x.shape[0]-1):
		for i in range(x.shape[1]-1):
			if hx[j,i] is not None:
				if hx[j+1,i] is not None:
					plt.plot([hi[j,i],hi[j+1,i]],[hj[j,i],hj[j+1,i]],'m',alpha=T.alpha(hy[j,i]))
		    		if hx[j,i+1] is not None:
					plt.plot([hi[j,i],hi[j,i+1]],[hj[j,i],hj[j,i+1]],'m',alpha=T.alpha(hy[j,i]))
				p=1 - 2*(j%2)
		    	if hx[j+1,i+p] is not None:
				plt.plot([hi[j,i],hi[j+1,i+p]],[hj[j,i],hj[j+1,i+p]],'m',alpha=T.alpha(hy[j,i]))

	# Draw circles
	for j in range(x.shape[0]):
	    for i in range(x.shape[1]):
		if hx[j,i] is not None:
		    mi,mj = T.polygon_ij(hx[j,i],hy[j,i],0.5*dd,nsegs=20)
		    plt.plot(mi,mj,'k',alpha=T.alpha(hy[j,i]))
	plt.xlim(0,T.ni-1); plt.ylim(T.nj-1,0);
	plt.axis('off');
	plt.savefig('Figures/hexaberg.png',dpi=300, bbox_inches='tight');
	




if __name__ == '__main__':

	#plot_2_panel_figure=True
	
	#if plot_2_panel_figure is True:
	#	plt.subplot(1,2,1)
	#	Schematic()
	#	plt.subplot(1,2,2)
	#	Perspective()
	#else:
	#	Perspective()
	Perspective()
	
	plt.show()
	print 'Script complete'
	#sys.exit(main())















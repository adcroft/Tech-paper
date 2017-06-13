#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy
#%matplotlib inline
import matplotlib.image as mpimg


def draw_polygon(x, y, r, nsegs=6, rot=numpy.pi/6, ls='k-'):
	a = numpy.linspace(-numpy.pi+rot,numpy.pi+rot,nsegs+1)
	plt.plot(x+r*numpy.cos(a),y+r*numpy.sin(a),ls)

##########################################################  Main Program   #########################################################################
####################################################################################################################################################

def Schematic():
	#Read in the photo
	im = mpimg.imread('Figures/hexaberg.png')

	plt.figure(figsize=(8,8));
	with plt.xkcd():
		# Ocean grid
		L = 5
		for x in numpy.arange(0,L+1,1): plt.plot([x,x],[0,L],'k:')
		for y in numpy.arange(0,L+1,1): plt.plot([0,L],[y,y],'k:')
		eps = 0.02; plt.xlim(-eps,L+eps); plt.ylim(-eps,L+eps);
		x,y = 0,3
		plt.gca().annotate('Ocean mesh', xy=(x+.05,y-0.05), xytext=(x+0.1,y-0.5),
				   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
		# Cluster of point bergs
		numpy.random.seed(3)
		x,y,dr = 1.4,3.7,.35
		plt.plot(x+dr*numpy.random.randn(15),y+dr*numpy.random.randn(15),'r.')
		plt.gca().annotate('(i) Non-interacting point bergs', xy=(x+.1,y+.1), xytext=(x-.9,y+.7))
		# Cluster of finite sized bergs
		x,y,dr = 2.5,2,2.5
		for n in range(10):
		    draw_polygon(x+dr*numpy.random.rand(),y+dr*numpy.random.rand(),
				 0.05+0.1*numpy.random.rand(),nsegs=20,ls='r')
		plt.gca().annotate('(ii) Finite extent bergs', xy=(x+1.5,y+1.5), xytext=(x+.5,y+2.1))
		plt.gca().annotate('Interacting bergs', xy=(x+1.3,y+.8), xytext=(x+.5,y+.3),
				   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
		# A tabular berg
		hx,hy=[1.5,0,1,2,0.5,1.5,2.5,-1,0,1,2,-.5,.5,1.5,0],[-1,0,0,0,1,1,1,2,2,2,2,3,3,3,4]
		x0,y0,dr,rot=2.2,0.8,0.3,20.*numpy.pi/180.
		for x,y in zip(hx,hy):
			dx,dy = x*dr,y*dr*numpy.sqrt(3/4.)
			dx,dy = numpy.cos(rot)*dx-numpy.sin(rot)*dy,numpy.cos(rot)*dy+numpy.sin(rot)*dx
			draw_polygon(x0+dx,y0+dy,0.45*dr,nsegs=20,ls='r')
			for x2,y2 in zip(hx,hy):
				r2 = (x-x2)**2+(y-y2)**2
				if r2>0 and r2<1.5 and y2>=y:
					dx2,dy2 = x2*dr,y2*dr*numpy.sqrt(3/4.)
					dx2,dy2 = numpy.cos(rot)*dx2-numpy.sin(rot)*dy2,numpy.cos(rot)*dy2+numpy.sin(rot)*dx2
					plt.plot([x0+dx,x0+dx2],[y0+dy,y0+dy2],'m')
		plt.gca().annotate('(iii) Tabular berg represented', xy=(x0-1.,y0+.8), xytext=(x0-0.8,y0-.5))
		plt.gca().annotate('by bonded elements', xy=(x0-1.,y0+.8), xytext=(x0-0.55,y0-.7))
		plt.gca().annotate('Bonds', xy=(x0+0.6,y0+.35), xytext=(x0+1.,y0+.7),
				   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2'))
		plt.axis('off');
		#ax = plt.gcf().add_axes([0.02,0.1,.55,.3]);
		#ax.imshow(im); plt.axis('off')

		plt.savefig('Figures/schematic_berg.png',dpi=300, bbox_inches='tight');




if __name__ == '__main__':
	Schematic()
	plt.show()
	print 'Script complete'
	#sys.exit(main())















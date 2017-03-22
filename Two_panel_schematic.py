#!/usr/bin/env python
import matplotlib.pyplot as plt
#import matplotlib
import numpy
#%matplotlib inline
import matplotlib.image as mpimg



##########################################################  Main Program   #########################################################################
####################################################################################################################################################





if __name__ == '__main__':
	
	#Read in image 1
	im_hexaberg = mpimg.imread('Figures/hexaberg.png')
	
	#Read in the photo
	im_schematic = mpimg.imread('Figures/schematic_berg.png')
	
	plt.figure(figsize=(12,8));	
	#ax=plt.subplot(1,2,1)
	ax = plt.gcf().add_axes([0.02,0.1,.6,.6]);
	ax.imshow(im_schematic); plt.axis('off')
	ax.text(1,1,'(a)', ha='right', va='bottom',transform=ax.transAxes,fontsize=15)
	#ax=plt.subplot(1,2,2)
	ax = plt.gcf().add_axes([0.5,0.15,.5,.5]);
	ax.imshow(im_hexaberg); plt.axis('off')
	ax.text(1,1,'(b)', ha='right', va='bottom',transform=ax.transAxes,fontsize=15)
		
	#plt.gca().annotate('Ocean mesh', xy=(x+.05,y-0.05), xytext=(x+0.1,y-0.5),
	#			   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

	plt.savefig('Figures/Two_panel_schematic_berg.png',dpi=300,bbox_inches='tight');
	
	#Schematic()
	plt.show()
	print 'Script complete'

	#sys.exit(main())















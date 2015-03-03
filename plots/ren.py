import numpy as np
import pdb
import scipy.io
import matplotlib.pyplot as plt

experiments = ['AZ_VARIED', 'EL_VARIED', 'LIGHT_AZ_VARIED']

for exp in experiments:

	#cosd plot
	gt_brun = scipy.io.loadmat(exp + '_COSD_brunelleschi.mat')['x']
	gt_dona = scipy.io.loadmat(exp + '_COSD_donatello.mat')['x']
	gt_inv = scipy.io.loadmat(exp + '_COSD_invariance_scaled_import_brunelleschi_invariance_strength_0.01_force_invariance.mat')['x']
	
	if exp == 'AZ_VARIED':
		xx = np.arange(-1.5,1.51,0.05)
		tit = 'Pose (Azimuth)'
	elif exp == 'EL_VARIED':
		xx = np.arange(-0.4,0.41,0.05)
		tit = 'Pose (Elevation)'
	else:
		xx = np.arange(-80,80.1,2)
		tit = 'Light'

	fig,ax = plt.subplots()
	plt.xlabel('Value', fontsize=25)
	plt.ylabel('Cosine Distance', fontsize=25)
	fig.suptitle(tit, fontsize=25)
	plt.plot(xx, gt_brun,c='blue', alpha=0.5, label='Without Shape Bias',linewidth=4)
	plt.plot(xx, gt_dona,c='red', alpha=0.5, label='With Shape Bias',linewidth=4)
	plt.plot(xx, gt_inv,c='gray', alpha=0.5, label='Without Shape Bias\n(Targeted Invariance)',linewidth=4)
	
	# legend = ax.legend(loc='upper right', shadow=True, fontsize=20)
	
	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0, box.width * 0, box.height])
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 25)

	plt.savefig(exp + '_cosd.pdf')
	# plt.close()	



	#correlation plot
	gt_brun = scipy.io.loadmat(exp + '_GT_brunelleschi.mat')['x']
	gt_dona = scipy.io.loadmat(exp + '_GT_donatello.mat')['x']
	gt_inv = scipy.io.loadmat(exp + '_GT_invariance_scaled_import_brunelleschi_invariance_strength_0.01_force_invariance.mat')['x']

	inf_brun = scipy.io.loadmat(exp + '_INF_brunelleschi.mat')['x']
	inf_dona = scipy.io.loadmat(exp + '_INF_donatello.mat')['x']
	inf_inv = scipy.io.loadmat(exp + '_INF_invariance_scaled_import_brunelleschi_invariance_strength_0.01_force_invariance.mat')['x']
	
	fig,ax = plt.subplots()
	plt.xlabel('Ground Truth', fontsize=25)
	plt.ylabel('Inferred', fontsize=25)
	fig.suptitle(tit, fontsize=25)
	plt.scatter(gt_brun, inf_brun,c='blue', alpha=0.4,label='Without Shape Bias')
	plt.scatter(gt_dona, inf_dona,c='red', alpha=0.4, label='With Shape Bias')
	plt.scatter(gt_inv, inf_inv,c='gray', alpha=0.4, label='Without Shape Bias\n(Targeted Invariance)')
	
	# legend = ax.legend(loc='upper right', shadow=True, fontsize=20)
	plt.savefig(exp + '_scatter.pdf')
	# plt.close()

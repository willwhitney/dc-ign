require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'rmsprop'
require 'KLDCriterion'
require 'LinearCR'
require 'Reparametrize'
require 'cutorch'
require 'cunn'
require 'optim' 
require 'GaussianCriterion'
require 'testf'
require 'utils'
require 'config'
require 'image'

opt = {}
opt.save = 'F96_H120_lr0_0005_BACK_3'
-- model = torch.load('F96_H120/vxnet.net')
model = init_network2_150()
parameters, gradients = model:getParameters()

print("Loading old weights!")
print(opt.save)
lowerboundlist = torch.load(opt.save .. '/lowerbound.t7')
lowerbound_test_list = torch.load(opt.save .. '/lowerbound_test.t7')
state = torch.load(opt.save .. '/state.t7')
p = torch.load(opt.save .. '/parameters.t7')
print('Loaded p size:', #p)
parameters:copy(p)
epoch = lowerboundlist:size(1)
config = torch.load(opt.save .. '/config.t7')


criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

criterion:cuda()
KLD:cuda()



function getfeatures(fname)
	-- fname = "/home/tejas/Documents/MIT/Picture/programs/graphics_programming/3DFace/data/pair_0_0.png"
	local im_tmp = image.load(fname)
	im = torch.zeros(1,im_tmp:size()[2],im_tmp:size()[3])
	im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
	newim = image.scale(im[1], 150 ,150)
	batch = torch.zeros(1,1,imwidth,imwidth)  
	batch[1]=newim
	model:forward(batch:cuda())
	-- ftrs = model:get(2).output:double()
	ftrs = model.modules[1].modules[11].modules[1].output:double()
	-- print(ftrs:size())
	return ftrs
end
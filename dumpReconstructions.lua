
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
require 'SelectiveOutputClamp'
require 'SelectiveGradientFilter'

UNSUP = true


opt = {}

if UNSUP then
	MODE_TEST = 'test'
	opt.save = 'F96_H120_lr0_0005_BACKUP7'
	model = init_network2_150()
else
	MODE_TEST = 'FT_test'
	opt.save = 'MV_lategradfilter_fixedindices_import_picasso_shape_bias_shape_bias_amount_100'
	model = init_network2_150_mv(200, 96)
	clamps = model:findModules('nn.SelectiveOutputClamp')
	gradFilters = model:findModules('nn.SelectiveGradientFilter')
	opt.num_test_batches_per_type = 350
	opt.datasetdir = 'DATASET/TRANSFORMATION_DATASET'
end

parameters, gradients = model:getParameters()
print('parameters ssize:', #parameters)

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

if UNSUP then
	testf(true)
else
	testf_MV(true)
end
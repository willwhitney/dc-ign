
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


MODE_TEST = 'test'
opt = {}
opt.save = 'F96_H120_lr0_0005_BACKUP6'
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


testf(true)

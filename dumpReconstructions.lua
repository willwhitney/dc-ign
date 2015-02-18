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
model = torch.load('log_NEW_init_network2_150_F40_H60/vxnet.net')

criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

criterion:cuda()
KLD:cuda()


testf(true)

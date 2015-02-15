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

fname = 'logs_init_network2_150'
model = torch.load(fname .. '/vxnet.net')
params, grad = model:getParameters()

torch.save(fname .. '/params.t7')
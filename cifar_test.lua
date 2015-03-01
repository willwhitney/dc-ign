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
require 'mattorch'
require 'train_svm'

opt = {}
opt.cuda = true
opt.save = 'logs_cifar'

trsize = 1000--50000
tesize = 1000--10000
trainData = {
  data = torch.Tensor(50000, 3072),
  labels = torch.Tensor(50000),
  size = function() return trsize end
}

for i = 0,4 do
  subset = mattorch.load('DATASET/cifar-10-batches-mat/data_batch_' .. (i+1) .. '.mat')
  trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
  trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end

trainData.labels = trainData.labels + 1
subset = mattorch.load('DATASET/cifar-10-batches-mat/test_batch.mat')
testData = {
  data = subset.data:t():double(),
  labels = subset.labels[1]:double(),
  size = function() return tesize end
}
testData.labels = testData.labels + 1
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]
testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

model = init_network2_color_width32()

criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

if opt.cuda then
    criterion:cuda()
    KLD:cuda()
    model:cuda()
end

parameters, gradients = model:getParameters()
print('Num before', #parameters)

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

print('Num of parameters:', #parameters)

ENC_SIZE = 200
opt.batchSize = 200

trainFeatures = torch.zeros(trsize,ENC_SIZE)
testFeatures = torch.zeros(tesize,ENC_SIZE)

function extract_features(ftrs, dataset)
	local globalcnt = 1

	for t =  1,dataset.data:size()[1],opt.batchSize do
		collectgarbage()
		-- create mini batch
		local raw_inputs = torch.zeros(opt.batchSize, 3* 32 *32)
		local cnt = 1
		for ii = t,math.min(t+opt.batchSize-1,dataset.data:size()[1]) do
			raw_inputs[cnt] = dataset.data[ii]
			cnt = cnt + 1
		end      
		raw_inputs = raw_inputs/255.0
		inputs = raw_inputs:cuda()
		model:forward(inputs)
		allftrs = model:get(2).output:double()
		for ii = 1, allftrs:size()[1] do
			ftrs[globalcnt] = allftrs[ii]
			globalcnt = globalcnt + 1
		end
		print(t, "/", dataset.data:size()[1])
	end
	return ftrs
end

function normalize_data(ftrs,sz)
	for i = 1,sz do
		local mean = ftrs[{i,{}}]:mean()
		local std = math.sqrt(ftrs[{i,{}}]:var()+0.01)
		ftrs[{i,{}}] = ftrs[{i,{}}]:add(-mean):div(std)
	end
end

trainFeatures = extract_features(trainFeatures, trainData)
-- testFeatures = extract_features(testFeatures, testData)

print("normalize data ...")
normalize_data(trainFeatures, trsize)


trainFeatures = trainFeatures:double()
testFeatures = testFeatures:double()

print("train SVM classifier ... ")
trainFeatures = torch.cat(trainFeatures, torch.ones(trainFeatures:size(1)), 2)
local theta = train_svm(trainFeatures, trainData.labels, 100);
local val,idx = torch.max(trainFeatures * theta, 2)
local match = torch.eq(trainData.labels, idx:float():squeeze()):sum()
local accuracy = match/trainData.labels:size(1)*100
print('train accuracy is '..accuracy..'%')


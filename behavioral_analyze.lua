-- take a trained model and output features for all the behavioral test images
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

model = torch.load('logs_init_network2_150/vxnet.net')


COLOR = false
imwidth = 150
batch = torch.zeros(96*2,1,imwidth,imwidth)  
cnt=1
for id=1,96 do
	for pid = 0,1 do
		fname = 'DATASET/images_wbg/' .. 'pair_' .. id .. '_' .. pid .. '.png'
		im_tmp = image.load(fname)

		if COLOR==true then
        	im = torch.zeros(3,150, 150)
        	if im:size()[2] ~= imwidth then
          		newim = image.scale(im, imwidth ,imwidth)
        	else
          		newim = im
        	end
      	else
        	im = torch.zeros(1,256, 256)
        	im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
        	im = image.scale(im[1], imwidth ,imwidth)
        	im = torch.reshape(im, 1, imwidth, imwidth)
        	batch[cnt] = im
        	cnt=cnt+1
      	end

	end
end

print(model)
res = model:forward(batch:cuda())

mean = model.modules[1].modules[11].modules[1].output:double()
sigma = model.modules[1].modules[11].modules[2].output:double()
-- mean = model.modules[3].modules[2].output:double()
print(mean:size())
require 'mattorch'
mattorch.save('analyzeBehavioral/mean.mat', mean)
mattorch.save('analyzeBehavioral/sigma.mat', sigma)














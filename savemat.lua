require 'image'
require 'mattorch'
bsize = 50
imwidth = 150


COLOR = false

TOTALFACES = 5230
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

function cache(id, mode)
    collectgarbage()
    batch = torch.load('DATASET/th_' .. mode .. '/batch' .. id)
    mattorch.save('thirdparty/' .. mode .. '/batch' .. id .. '.mat', batch:double())
end

for t = 3000, num_train_batches do
	cache(t, 'training')
	print(t)
end

-- for t = 1, num_test_batches do
-- 	cache(t, 'test')
-- 	print(t)
-- end

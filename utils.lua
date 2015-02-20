
function load_batch(id, mode)
  -- return torch.load('DATASET/th_' .. mode .. '/batch' .. id)
  return torch.load('/om/user/tejask/facegen/DATASET/th_' .. mode .. '/batch' .. id)
end

function getLowerbound(data)
    local lowerbound = 0
    N_data = num_test_batches
    for i = 1, N_data, batchSize do
        local batch = data[{{i,i+batchSize-1},{}}]
        local f = model:forward(batch)
        local target = target or batch.new()
        target:resizeAs(f):copy(batch)
        local err = - criterion:forward(f, target)

        local encoder_output = model:get(1).output

        local KLDerr = KLD:forward(encoder_output, target)

        lowerbound = lowerbound + err + KLDerr
    end
    return lowerbound
end

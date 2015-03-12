
function load_batch(id, mode)
  res = torch.load('DATASET/th_' .. mode .. '/batch' .. id)
  return res
end

function load_mv_batch(id, dataset_name, mode)
    return torch.load(opt.datasetdir .. '/th_' .. dataset_name .. '/' .. mode .. '/batch' .. id)
end

function load_random_mv_batch(mode)
    local variation_type = math.random(4)
    local variation_name = ""
    if variation_type == 1 then
        variation_name = "AZ_VARIED"
    elseif variation_type == 2 then
        variation_name = "EL_VARIED"
    elseif variation_type == 3 then
        variation_name = "LIGHT_AZ_VARIED"
    elseif variation_type == 4 then
        variation_name = "SHAPE_VARIED"
    end

    id = 1
    if mode == MODE_TRAINING then
        id = math.random(opt.num_train_batches_per_type)
    elseif mode == MODE_TEST then
        id = math.random(opt.num_test_batches_per_type)
    end
    return load_mv_batch(id, variation_name, mode), variation_type
end

-- has a bias towards shape samples
function load_random_mv_shape_bias_batch(mode)
    local variation_type = math.random(4 + opt.shape_bias_amount)
    local variation_name = ""
    if variation_type == 1 then
        variation_name = "AZ_VARIED"
    elseif variation_type == 2 then
        variation_name = "EL_VARIED"
    elseif variation_type == 3 then
        variation_name = "LIGHT_AZ_VARIED"
    else
        variation_name = "SHAPE_VARIED"
        variation_type = 4
    end

    id = 1
    if mode == MODE_TRAINING then
        id = math.random(opt.num_train_batches_per_type)
    elseif mode == MODE_TEST then
        id = math.random(opt.num_test_batches_per_type)
    end
    return load_mv_batch(id, variation_name, mode), variation_type
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


function load_batch(id, mode)
    -- return torch.load('DATASET/th_' .. mode .. '/batch' .. id)
    return torch.load('/om/user/tejask/facegen/DATASET/th_' .. mode .. '/batch' .. id)
end

function load_mv_batch(id, dataset_name, mode)
    return torch.load('DATASET/TRANSFORMATION_DATASET/th_' .. dataset_name .. '/' .. mode .. '/batch' .. id)
end

function load_random_mv_batch(mode)
    local variation_type = math.random(3)
    local variation_name = ""
    if variation_type == 1 then
        variation_name = "AZ_VARIED"
    elseif variation_type == 2 then
        variation_name = "EL_VARIED"
    elseif variation_type == 3 then
        variation_name = "LIGHT_AZ_VARIED"
    end

    id = 1
    if mode == MODE_TRAINING then
        id = math.random(350)
    elseif mode == MODE_TEST then
        id = math.random(30)
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

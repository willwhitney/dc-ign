-- test function
function testf(saveAll)
  -- in case it didn't already exist
  os.execute('mkdir ' .. 'tmp')

   -- local vars
   local time = sys.clock()
   -- test over given dataset
   print('<trainer> on testing Set:')
   reconstruction = 0
   local lowerbound = 0
   for t = 1, num_test_batches do
      collectgarbage()
      -- create mini batch
      local raw_inputs = load_batch(t, MODE_TEST)
      local targets = raw_inputs

      inputs = raw_inputs:cuda()
      -- disp progress
      xlua.progress(t, num_test_batches)

      -- test samples
      local preds = model:forward(inputs)

      local f = preds
      local target = targets
      local err = - criterion:forward(f, target:cuda())
      local encoder_output = model:get(1).output
      local KLDerr = KLD:forward(encoder_output, target)
      lowerbound = lowerbound + err + KLDerr


      preds = preds:float()

      reconstruction = reconstruction + torch.sum(torch.pow(preds-targets,2))

      if saveAll then
        torch.save('tmp/preds' .. t, preds)
      else
        if t == 1 then
            torch.save('tmp/preds' .. t, preds)
        end
      end
   end
   -- timing
   time = sys.clock() - time
   time = time / num_test_batches
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   reconstruction = reconstruction / (bsize * num_test_batches * 3 * 150 * 150)
   print('mean MSE error (test set)', reconstruction)
   testLogger:add{['% mean class accuracy (test set)'] = reconstruction}
   reconstruction = 0
   return lowerbound
end

-- test function for monovariant tests
function testf_MV(saveAll)
  -- in case it didn't already exist
  os.execute('mkdir -p' .. 'tmp')

  -- local vars
  local time = sys.clock()
  -- test over given dataset
  print('<trainer> on testing Set:')
  reconstruction = 0
  local lowerbound = 0

  -- turn the clamps off so we get full batch outputs
  for clampIndex = 1, #clamps do
    clamps[clampIndex].active = false
    gradFilters[clampIndex].active = false
  end

  for _, dataset_name in pairs({"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"}) do
    local save_dir = 'tmp' .. '/' .. opt.save .. '/' .. dataset_name
    os.execute('mkdir -p ' .. save_dir)

     for t = 1, opt.num_test_batches_per_type do
        collectgarbage()
        -- create mini batch
        local raw_inputs = load_mv_batch(t, dataset_name, MODE_TEST)
        local targets = raw_inputs

        inputs = raw_inputs:cuda()
        -- disp progress
        xlua.progress(t, opt.num_test_batches_per_type)

        -- test samples
        local preds = model:forward(inputs)

        local f = preds
        local target = targets
        local err = - criterion:forward(f, target:cuda())
        local encoder_output = model:get(1).output
        local KLDerr = KLD:forward(encoder_output, target)
        lowerbound = lowerbound + err + KLDerr


        preds = preds:float()

        reconstruction = reconstruction + torch.sum(torch.pow(preds-targets,2))

        if saveAll then
          torch.save(save_dir..'/preds' .. t, preds)
        else
          if t == 1 then
              torch.save(save_dir..'/preds' .. t, preds)
          end
        end
     end
   end

   -- timing
   time = sys.clock() - time
   time = time / opt.num_test_batches
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   reconstruction = reconstruction / (bsize * opt.num_test_batches * 3 * 150 * 150)
   print('mean MSE error (test set)', reconstruction)
   testLogger:add{['% mean class accuracy (test set)'] = reconstruction}
   reconstruction = 0
   return lowerbound
end


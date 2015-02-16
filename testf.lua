-- test function
function testf(saveAll)
   -- local vars
   local time = sys.clock()
   -- test over given dataset
   print('<trainer> on testing Set:')
   reconstruction = 0
   local lowerbound = 0
   for t = 1, num_test_batches do
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

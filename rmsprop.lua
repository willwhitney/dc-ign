--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.learningRateDecay' : learning rate decay
- 'config.weightDecay'       : weight decay
- 'config.momentum'          : momentum
- 'config.dampening'         : dampening for momentum
- 'config.nesterov'          : enables Nesterov momentum
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
- 'state.rms'                 : vector of individual learning rates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function rmsprop(opfunc, x, config, state)
   -- get parameters
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local b1 = config.momentumDecay or 1
   local b2 = config.updateDecay or 1

   local fx, dfdx = opfunc(x)

   state.evalCounter = state.evalCounter or 0
   state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   state.v = state.v or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   
   -- Decay term
   state.m:mul(1 - b1)

   -- New term 
   state.momentum_dfdx = state.momentum_dfdx or torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
   state.momentum_dfdx:copy(dfdx)

   state.m:add(state.momentum_dfdx:mul(b1))

   -- Decay term of update
   state.v:mul(1 - b2)

   -- New update
   dfdx:cmul(dfdx):mul(b2)

   state.v:add(dfdx)

   -- calculate update step
   state.evalCounter = state.evalCounter + 1

   --Create new momentum Tensors for cutorch compatibility
   state.momentum_update = state.momentum_update or torch.Tensor():typeAs(state.m):resizeAs(state.m)
   state.momentum_update:copy(state.m)

   state.update = state.update or torch.Tensor():typeAs(state.v):resizeAs(state.v)
   state.update:copy(state.v)

   state.momentum_update:cdiv(state.update:add(1e-8):sqrt())

   local gamma = (math.sqrt(1 - math.pow(1 - b2,state.evalCounter))/(1 - math.pow(1 - b1, state.evalCounter)))
   state.momentum_update:mul(gamma)

   x:add(-lr, state.momentum_update)

   -- return x*, f(x) before optimization
   return x,{fx}
end




-- function rmsprop( x, dfdx, config, gradAverage)
-- 	meta_learning_alpha = 0.005--1e-3

-- 	gradAverageArr=torch.zeros(3)
-- 	gamma = {math.exp(1), math.exp(3),math.exp(6)} 
-- 	for i=1,3 do
--     	gradAverageArr[i] = 1/gamma[i] * torch.pow(dfdx:norm(), 2) + (1-(1/gamma[i]))*gradAverage
--     end
--     gradAverage = torch.max(gradAverageArr)
--     x = x - (dfdx*meta_learning_alpha / gradAverage)
--     return x

-- 	-- local config = config or {}
-- 	-- local lr = config.learningRate or 1e-3
-- 	-- local b1 = config.momentumDecay or 1
-- 	-- local b2 = config.updateDecay or 1

-- 	-- config.evalCounter = config.evalCounter or 0
-- 	-- config.m = config.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
-- 	-- config.v = config.v or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
	
-- 	-- -- Decay term
-- 	-- config.m:mul(1 - b1)

-- 	-- config.momentum_dfdx = config.momentum_dfdx or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
-- 	-- config.momentum_dfdx:copy(dfdx)

-- 	-- config.m:add(config.momentum_dfdx:mul(b1))


-- 	-- -- Decay term of update
-- 	-- config.v:mul(1 - b2)

-- 	-- -- New update
-- 	-- dfdx:cmul(dfdx):mul(b2)

-- 	-- config.v:add(dfdx)

-- 	-- -- calculate update step
-- 	-- config.evalCounter = config.evalCounter + 1
-- 	-- --Create new momentum Tensors for cutorch compatibility
-- 	-- config.momentum_update = config.momentum_update or torch.Tensor():typeAs(config.m):resizeAs(config.m)
-- 	-- config.momentum_update:copy(config.m)
-- 	-- config.update = config.update or torch.Tensor():typeAs(config.v):resizeAs(config.v)
-- 	-- config.update:copy(config.v)
-- 	-- config.momentum_update:cdiv(config.update:add(1e-8):sqrt())
	
-- 	-- local gamma = (math.sqrt(1 - math.pow(1 - b2,config.evalCounter))/(1 - math.pow(1 - b1, config.evalCounter)))
-- 	-- config.momentum_update:mul(gamma)
-- 	-- x:add(-lr, config.momentum_update)
-- 	-- return x
-- end

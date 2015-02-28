-- 
-- An Analysis of Single-Layer Networks in Unsupervised Feature Learning
-- by Adam Coates et al. 2011
--
-- The original MatLab code can be found in http://www.cs.stanford.edu/~acoates/
-- Tranlated to Lua/Torch7
--
require("optim")

function train_svm(data, label, c, verbose)
   local nclass = torch.max(label)
   local w      = torch.zeros(data:size(2)*nclass)

   local function svm_loss(w)
      local X = data
      local m = X:size(1)
      local n = X:size(2)
      local l = label
      local k = nclass
      local c = 100
      local theta = w:reshape(n, k)

      local y = l:reshape(m, 1):expand(m, k)
      local ypos = torch.range(1, k):resize(1, k):expand(m, k)
      local Y = torch.eq(y,ypos):float():mul(2):add(-1)

      -- max{0, 1-y*W*X}
      -- print(torch.type(X), torch.type(Y))
      -- X = X:float()
      -- print(theta:size(), X:size())
      local tmp0 = (X*theta)
      -- print(tmp0:size())
      tmp0 = tmp0:float()
      local Z = tmp0:cmul(-Y):add(1)
      -- print(Z)
      local tmp1 = torch.gt(Z,0):float()
      local margin = tmp1:cmul(Z)

      -- gradient = theta - 2*c/m * (X' * (margin .* Y))

      local tmp3 = X:t()
      Y = Y:float()
      tmp3 = tmp3:float()
      local tmp2 = (tmp3*Y)
      print(margin:size(), tmp2:size(), tmp3:size(), Y:size())
      tmp2 = (tmp2:cmul(margin)):float()
      local g = tmp2:mul(-2*c/m):add(theta)

      -- loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2)
      local loss = 0.5*theta:pow(2):sum()+c*margin:pow(2):mean(1):sum()

      return loss, g:resize(n*k)
   end

   local w,fw,i = optim.lbfgs(svm_loss, w, {maxIter=1000, maxEval=1000})
   if verbose then for i=1,#fw do print(i,fw[i]); end end
   return w:resize(data:size(2), nclass)
end


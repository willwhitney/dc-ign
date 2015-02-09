
function rmsprop( x, dfdx, gradAverage)
	meta_learning_alpha = 0.005

	gradAverageArr=torch.zeros(3)
	gamma = {math.exp(1), math.exp(3),math.exp(6)} 
	for i=1,3 do
    	gradAverageArr[i] = 1/gamma[i] * torch.pow(dfdx:norm(), 2) + (1-(1/gamma[i]))*gradAverage
    end
    gradAverage = torch.max(gradAverageArr)
    x = x - (dfdx*meta_learning_alpha / gradAverage)
    return x
	
end

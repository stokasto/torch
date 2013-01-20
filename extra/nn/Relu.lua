local Relu, parent = torch.class('nn.Relu', 'nn.Module')

function Relu:__init(lam, neg)
   parent.__init(self)
   self.lambda = lam or 0
   self.neg = neg or false
end

function Relu:updateOutput(input)
   input.nn.Relu_updateOutput(self, input)
   return self.output
end

function Relu:updateGradInput(input, gradOutput)
   input.nn.Relu_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

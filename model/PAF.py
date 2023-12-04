import torch
class MyActFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        one = torch.ones_like(input_)
        zero = torch.zeros_like(input_)
        x1 = torch.where(input_ <= -1, ((0 * input_ + 0.0848) * input_ + 0.1820) * input_  -0.0278 * one, zero)
        x2 = torch.where((input_ > -1) & (input_ <= -0.45), ((-0.0952 * input_ -0.2149) * input_ -0.1319) * input_ -0.1371*one, zero)
        x3 = torch.where((input_ > -0.45) & (input_ <= 0), ((1.4094 * input_ +1.8162) * input_ + 0.7821 ) * input_ + 0*one, zero)
        x4 = torch.where((input_ > 0) & (input_ <= 0.3),((-2.4505 * input_ + 1.8162) * input_ + 0.7821 ) * input_ +0*one, zero)
        x5 = torch.where((input_ > 0.3) & (input_ <= 1),((0.2006 * input_ -0.6890) * input_ + 1.5713) * input_ -0.0829 * one, zero)
        x6 = torch.where(input_ > 1, ((0 * input_ -0.0873) * input_ + 0.9696) * input_ + 0.1177 * one, zero)
        output = x1 + x2 + x3 + x4 + x5 + x6
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        one = torch.ones_like(input_)
        zero = torch.zeros_like(input_)
        grad_input_1 = torch.where(input_ <= -1, 0 * 3 * input_ ** 2 + 0.0848 * 2 * input_ +  0.1820 * one, zero)
        grad_input_2 = torch.where((input_ > -1) & (input_ <= -0.45),-0.0952 * 3 * input_ ** 2 -0.2149 * 2 * input_ -0.1319 * one, zero)
        grad_input_3 = torch.where((input_ > -0.45) & (input_ <= 0),1.4094 * 3 * input_ ** 2 + 1.8162 * 2 * input_ + 0.7821 * one, zero)
        grad_input_4 = torch.where((input_ > 0) & (input_ <= 0.3),-2.4505 * 3 * input_ ** 2 + 1.8162 * 2 * input_ + 0.7821 * one, zero)
        grad_input_5 = torch.where((input_ > 0.3) & (input_ <= 1),0.2006 * 3 * input_ ** 2 -0.6890 * 2 * input_ + 1.5713 * one, zero)
        grad_input_6 = torch.where(input_ > 1,0 * 3 * input_ ** 2 -0.0873 * 2 * input_ + 0.9696 * one, zero)
        output = grad_input_1 + grad_input_2 + grad_input_3 + grad_input_4 + grad_input_5 + grad_input_6
        return grad_input * output
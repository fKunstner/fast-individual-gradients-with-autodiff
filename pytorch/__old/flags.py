import torch 

HACKFLAG = 'hacked_to_avoid_summation_I_am_really_sorry_hackmode'
LEAFFLAG = 'hacked_to_avoid_summation_I_am_really_sorry_leafflag'

def is_hackmode(grad_tensor):
	return getattr(grad_tensor, HACKFLAG, False)
def is_leafnode(grad_tensor):
	return getattr(grad_tensor, LEAFFLAG, False)

def set_hackmode(grad_tensor):
	setattr(grad_tensor, HACKFLAG, True)
	
def set_leafnode(grad_tensor):
	setattr(grad_tensor, LEAFFLAG, True)
	
def hacked_starting_point():
	starting_point = torch.tensor(1.0)
	set_hackmode(starting_point)
	return starting_point
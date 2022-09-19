import torch
import torch.nn  as nn

"""
Created on Sunday 22 Mar 2020
@authors: Alan Preciado, Santosh Muthireddy
"""
def DDC_loss(source_activation, target_activation):
	"""
	From the paper, the loss used is the maximum mean discrepancy (MMD)
	:param source: torch tensor: source data (Ds) with dimensions DxNs
	:param target: torch tensor: target data (Dt) with dimensons DxNt
	"""

	diff_domains = source_activation - target_activation
	loss = torch.mean(torch.mm(diff_domains, torch.transpose(diff_domains, 0, 1)))

	return loss

class mix_mmdaloss(nn.Module):
	def __init__(self,batch_size=64):
		super(mix_mmdaloss,self).__init__()
		self.bs = batch_size
	def forward(self,feas,lam1=1.0,lam2=1.0,atten=None):
		num_sam = feas.size()[0]
		n = int(num_sam/int(self.bs))
		from pdb import set_trace
		if n==2:
			feas_s,feas_t = torch.chunk(feas, n, dim=0)

			mmd_loss = DDC_loss(feas_s,feas_t)
			return mmd_loss
		elif n==3:
			feas_s,feas_mix,feas_t = torch.chunk(feas, n, dim=0)
			if atten is not None:
				feas_s = torch.mul(feas_s,atten[:,0].unsqueeze(-1))
				feas_t = torch.mul(feas_t,atten[:,1].unsqueeze(-1))
			mmd_ds = DDC_loss(feas_s,feas_mix)
			mmd_dt = DDC_loss(feas_t,feas_mix)



			return lam1*mmd_ds+lam2*mmd_dt
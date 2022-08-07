import matplotlib.pyplot as plt
from collections import defaultdict
max_num_epochs = 100
warm_up_epochs = 5
lr_milestones = [20,40]
# MultiStepLR without warm up
multistep_lr = lambda epoch: 0.1**len([m for m in lr_milestones if m <= epoch])
# warm_up_with_multistep_lr
warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
# warm_up_with_step_lr
gamma = 0.1; stepsize = 20
warm_up_with_step_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else gamma**((epoch - warm_up_epochs)//stepsize)
# warm_up_with_cosine_lr
warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_num_epochs - warm_up_epochs) * math.pi) + 1)
epochs =[]
lrs = []
baselr = 0.0035
for epoch in range(max_num_epochs):
    lr = baselr * warm_up_with_step_lr(epoch)
    epochs.append(epoch)
    lrs.append(lr)

plt.plot(epochs,lrs)
plt.show()
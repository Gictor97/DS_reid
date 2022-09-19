from __future__ import print_function, absolute_import
import time
import torch
from .utils.meters import AverageMeter
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, TripletLossXBM, DivLoss, BridgeFeatLoss, BridgeProbLoss,domain_regluarization_loss,mix_mmdaloss
from pdb import set_trace

__all ={'Base_trainer',"idm_trainer"}
class Base_trainer(object):
    def __init__(self,arg,model,num_classes,xbm=None,margin=None):
        super(Base_trainer,self).__init__()
        self.model = model
        self.arg = arg
        self.num_classes = num_classes
        self.margin =margin
        self.loss_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.loss_tri = TripletLoss(self.margin).cuda()
        self.loss_tri_xbm = TripletLossXBM(self.margin).cuda()
        self.xbm = xbm
        self.loss_re = domain_regluarization_loss()
        self.losslist= list()
        self.losslist_ce =list()
        self.losslist_xbm = list()
    def train(self,epoch,data_source_loader,data_target_loader,optim,s_class,t_class,iters,gamma =0.0,xbm = None,print_freq=50):


        self.model.train()
        self.loss_ce = CrossEntropyLabelSmooth(s_class,t_class)
        #time
        batch_time = AverageMeter()
        data_time = AverageMeter()

        #loss
        losses = AverageMeter()
        loss_cees = AverageMeter()
        loss_tries = AverageMeter()
        loss_mmd = AverageMeter()
        #precision
        precision_ss = AverageMeter()
        precision_ts = AverageMeter()

        end = time.time()
        ##start iter
        for i in  range(iters):
            sour_data_batch = data_source_loader.next()
            tar_data_batch = data_target_loader.next()

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(sour_data_batch)
            t_inputs, t_targets, t_indexes = self._parse_data(tar_data_batch)
            data_time.update(time.time()-end)
            B,C,H,W = s_inputs.size()


            #create jointly data and target
            inputs = torch.cat((s_inputs,t_inputs),0)
            targets = torch.cat((s_targets,t_targets),0)


            # forward
            probs, feas= self.model(inputs)
            probs = probs[:, 0:s_class+t_class]
            feas_s,feas_t = torch.chunk(feas,2,dim=0)

            mmdloss = DDC_loss(feas_s.detach(),feas_t.detach())
            loss_ce = self.loss_ce(probs,targets)
            loss_tri = self.loss_tri(feas,targets)
            loss_re = self.loss_re(s_class,t_class,targets,probs)
            # enqueue and dequeue for xbm
            if self.xbm:
                loss_xbmes = AverageMeter()
                self.xbm.enqueue_dequeue(feas.detach(), targets.detach())
                xbm_feats, xbm_targets = self.xbm.get()
                loss_xbm = self.loss_tri_xbm(feas, targets, xbm_feats, xbm_targets)
                loss_xbmes.update(loss_xbm.item())
                self.losslist_xbm.append(loss_xbm.item())
                loss = loss_ce + loss_tri + loss_xbm
            else:
                loss = loss_ce+ loss_tri + gamma * mmdloss

            #backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            #calculate precision
            assert probs.size(0)%2==0
            prob_s,prob_t = torch.split(probs,probs.size(0)//2)
            prob_s,prob_t = prob_s.contiguous(),prob_t.contiguous()
            prec_s = accuracy(prob_s.detach(),s_targets.detach())
            prec_t = accuracy(prob_t.detach(),t_targets.detach())

            ##update contain
            losses.update(loss.item())
            loss_cees.update(loss_ce.item())
            loss_tries.update(loss_re.item())
            loss_mmd.update(mmdloss.item())
            precision_ss.update(prec_s[0].item())
            precision_ts.update(prec_t[0].item())
            self.losslist.append(loss.item())
            self.losslist_ce.append(loss_ce.item())

            batch_time.update(time.time() - end)
            end = time.time()
            # print log
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Loss_mmd {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      'Prec_t {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_target_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              loss_cees.val, loss_cees.avg,
                              loss_tries.val, loss_tries.avg,
                              loss_mmd.val,loss_mmd.avg,
                              precision_ss.val, precision_ss.avg,
                              precision_ts.val, precision_ts.avg
                              ))
                if self.xbm:
                    print('Loss_xbm{:.3f} ({:.3f})'
                          .format(loss_xbmes.val, loss_xbmes.avg))
        print(loss_re.item())
        if self.arg.dataset_target =='msmt17':
            print(loss_re.item())


    def _parse_data(self, inputs):
            imgs, _, pids, _, indexes = inputs
            return imgs.cuda(), pids.cuda(), indexes.cuda()


class idm_trainer(Base_trainer):
    def __init__(self,args,model,num_classes,mu1=1.0,mu2=1.0,mu3=1.0,a1=0,a2=0,xbm = None,margin=None):
        super(Base_trainer,self).__init__()
        self.model = model
        self.mu1 = mu1
        self.mu2 = mu2
        self.a1=a1
        self.a2=a2
        self.mu3 = float(mu3)
        self.num_classes = num_classes
        self.margin =margin
       # self.loss_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.loss_tri = TripletLoss(self.margin).cuda()
        self.loss_tri_xbm = TripletLossXBM(self.margin).cuda()
        self.loss_dvi = DivLoss()
        self.briloss_fea = BridgeFeatLoss().cuda()
        self.loss_ce = BridgeProbLoss(self.num_classes).cuda()
        self.loss_re = domain_regluarization_loss()
        self.loss_mmd = mix_mmdaloss(args.batch_size)
        self.xbm = xbm
        self.losslist = list()
        self.losslist_re = list()
    def train(self,epoch,stage,data_source_loader,data_target_loader,optim,s_class,t_class,iters,gamma,print_freq=100):

        self.model.train()
        #self.loss_ce = CrossEntropyLabelSmooth(s_class+t_class)
        self.loss_ce = BridgeProbLoss(s_class+t_class).cuda()
        #time
        batch_time = AverageMeter()
        data_time = AverageMeter()

        #loss
        losses = AverageMeter()
        loss_cees = AverageMeter()
        loss_tries = AverageMeter()
        loss_xbmes = AverageMeter()
        loss_bridge_proes = AverageMeter()
        loss_brifeaes = AverageMeter()
        loss_dives = AverageMeter()
        loss_mmdes =AverageMeter()
        loss_rees = AverageMeter()
        #precision
        precision_ss = AverageMeter()
        precision_ts = AverageMeter()
        end = time.time()

        ##start iter
        for i in  range(iters):
            sour_data_batch = data_source_loader.next()
            tar_data_batch = data_target_loader.next()

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(sour_data_batch)
            t_inputs, t_targets, t_indexes = self._parse_data(tar_data_batch)
            data_time.update(time.time()-end)

            B,C,H,W = s_inputs.size()

            #create jointly data and target data
            inputs = torch.cat((s_inputs,t_inputs),0)
            targets = torch.cat((s_targets,t_targets),0)

            #forward
            probs,feas,atten_lam=self.model(inputs,stage) # (192,class)(192,2048)()
            prob = probs[:,0:s_class+t_class]


       ##calculate
            ##split feature
            feas_s,feas_mixed,feas_t = torch.chunk(feas,3,dim=0)

            feas_s = feas_s.view(-1,feas_s.size(-1))
            feas_t = feas_s.view(-1, feas_s.size(-1))
            feas_ori = torch.cat((feas_s,feas_t),0).view(-1,feas_s.size(-1))
            feas_mixed = feas_s.view(-1, feas_s.size(-1))
            ##loss
            loss_div = self.loss_dvi(atten_lam)
            loss_tri = self.loss_tri(feas_ori, targets)
            loss_brifea = self.briloss_fea(feas_s,feas_t,feas_mixed,atten_lam)
            loss_ce,loss_bridge_pro = self.loss_ce(prob,targets,atten_lam[:,0].detach())
            loss_re  = self.loss_re(s_class,t_class,targets,probs,idm_lam =atten_lam)
            loss_mmd = self.loss_mmd(feas)

            self.losslist_re.append(loss_re.item())

            # enqueue and dequeue for xbm
            if self.xbm:
                self.xbm.enqueue_dequeue(feas_ori.detach(), targets.detach())
                xbm_feats, xbm_targets = self.xbm.get()
                loss_xbm = self.criterion_tri_xbm(feas_ori, targets, xbm_feats, xbm_targets)
                loss_xbmes.update(loss_xbm.item())
                loss = (1.0- self.mu1)*loss_ce + loss_tri+loss_xbm + self.mu1 * loss_bridge_pro +\
                    +self.mu2 *loss_brifea+self.mu3*self.loss_dvi+gamma*loss_re
            else:
                loss = (1.0 - self.mu1) * loss_ce + loss_tri  + self.mu1 * loss_bridge_pro + \
                       +self.mu2 * loss_brifea + self.mu3 * loss_div +self.a1* loss_mmd+ self.a2*loss_re
            #backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            #calculate precision
            assert prob.size(0)%3==0

            prob_s,prob_t,_ = torch.split(prob,prob.size(0)//3)
            prob_s,prob_t = prob_s.contiguous().view(-1,prob.size(-1)),prob_t.contiguous().view(-1,prob.size(-1))
            prec_s = accuracy(prob_s.detach(),s_targets.detach())
            prec_t = accuracy(prob_t.detach(),t_targets.detach())


            losses.update(loss.item())
            self.losslist.append(loss.item())
            loss_cees.update(loss_ce.item())
            loss_tries.update(loss_tri.item())
            loss_dives.update(loss_div.item())
            loss_brifeaes.update(loss_brifea.item())
            loss_bridge_proes.update(loss_bridge_pro.item())
            loss_rees.update(loss_re.item())
            loss_mmdes.update(loss_mmd.item())

            if self.xbm :
                loss_xbmes.update(loss_xbm.item())

            precision_ss.update(prec_s[0].item())
            precision_ts.update(prec_t[0].item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                if self.xbm:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_xbm {:.3f} ({:.3f}) '
                          'Loss_bridge_pro {:.3f}({:.3f})'
                          'Loss_brifea {:.3f}({:.3f})'
                          'Loss_bridge_pro {:.3f}({:.3f})'
                          'Loss_div {:.3f}({:.3f})'
                          'Loss_mmd {:.3f}({:.3f})'
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_target_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  loss_cees.val, loss_cees.avg,
                                  loss_tries.val, loss_tries.avg,
                                  loss_xbmes.val, loss_xbmes.avg,
                                  loss_bridge_proes.val, loss_bridge_proes.avg,
                                  loss_brifeaes.val, loss_brifeaes.avg,
                                  loss_dives.val, loss_dives.avg,
                                  loss_mmdes.val, loss_mmdes.avg,
                                  precision_ss.val, precision_ss.avg,
                                  precision_ts.val, precision_ts.avg
                                  ))
                else:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_bridge_pro {:.3f}({:.3f})'
                          'Loss_brifea {:.3f}({:.3f})'
                          'Loss_div {:.3f}({:.3f})'
                          'Loss_re {:.3f}({:.3f})'
                          'Loss_mmd {:.3f}({:.3f})'
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_target_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  loss_cees.val, loss_cees.avg,
                                  loss_tries.val, loss_tries.avg,
                                  loss_bridge_proes.val, loss_bridge_proes.avg,
                                  loss_brifeaes.val, loss_brifeaes.avg,
                                  loss_dives.val, loss_dives.avg,
                                  loss_rees.val, loss_rees.avg,
                                  loss_mmdes.val, loss_mmdes.avg,
                                  precision_ss.val, precision_ss.avg,
                                  precision_ts.val, precision_ts.avg
                                  ))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from pdb import set_trace
#myfont = fm.FontProperties(fname=r'/home/fs/anaconda3/envs/py/lib/python3.8/site-packages/matplotlib/mpl-data/SimHei.ttf')
#myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')


def draw(loss,loss_ce,loss_re,path):
    plt.subplot(3, 1, 1)
    draw_loss(loss, loss_ce)
    plt.subplot(3,1,2)
    draw_re(loss_re)
    plt.savefig(path+'/curve')

def draw_idm(loss,loss_tri,loss_xbm,path):
    plt.subplot(2, 1, 1)
    draw_loss(loss, loss_tri,loss_xbm)
    plt.subplot(2, 1, 2)
    learning_rate(lrlist,len(lrlist))
    plt.savefig(path+'/curve')

def draw_re(loss_re):
    plt.plot(loss_re,'b-',label = 'loss_re',linewidth=2)
    plt.xlabel(u'iter')
    plt.ylabel(u'loss')
    plt.legend()

def draw_loss(loss,loss_tri,loss_xbm=None ,length=None):

    # 获取字体，为设置中文显示
    plt.plot(loss,'r-',label='loss',linewidth=2)
    plt.plot(loss_tri,'k-',label = 'lossce',linewidth=2)
    if not loss_xbm:
        plt.plot(loss_xbm,'y-',label='loss_xbm',linewidth=2)
    plt.xlabel(u'iter')
    plt.ylabel(u'loss')
    # plt.xlabel(u"iter", fontproperties=myfont)
    # plt.ylabel(u"loss", fontproperties=myfolnt)
    plt.legend()  #

def learning_rate(lrlist,epoch):

      # 获取字体，为设置中文显示
    plt.plot(lrlist, 'r-', label='learing rate',linewidth=2)
    plt.xlabel(u'epoch')
    plt.ylabel(u'learning rate')
    # plt.xlabel(u"epoch", fontproperties=myfont)
    # plt.ylabel(u"learning rate", fontproperties=myfont)
    plt.xlim(0,epoch)
    plt.legend()  #


# myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf') #获取字体，为设置中文显示
# x = [1,2,3,4,5,6]
# data1 = [1,1.3,1.39,1.41,1.42,1.40]
# data2 = [1,1.36,1.55,1.70,1.78,1.82]
# data3 = [1,1.6,2.25,3.0,3.6,4.2]
# data4 = [1,1.8,2.5,3.1,3.8,4.5]
# y = [1,2,3,4,5,6]
#
# #使用plot方法绘制图形，marker表示图形节点处的显示，color设置颜色，label设置图示标签
# plt.plot(x,data1,'pk-.',label='data1')
# plt.plot(x,data2,marker="o",color='k',label='data2')
# plt.plot(x,data3,marker="*",color='k',label='data3')
# plt.plot(x,data4,marker="s",color='k',label='data4')
# plt.plot(x,y,marker="^",color='k',label=u'理想加速比')
#
# #设置x轴 y轴的标签，注意中文显示
# plt.xlabel(u"计算节点",fontproperties=myfont)
# plt.ylabel(u"加速比",fontproperties=myfont)
# plt.title("Title")
# #设置坐标轴值范围
# plt.xlim(1,6)
# plt.ylim(0,6)
#
# #最后这两句是显示图形
# plt.legend(prop=myfont)  #
# plt.show()


if __name__ =='__main__':
     a = torch.Tensor(2048)
     b = torch.Tensor(1024)
     c = torch.Tensor(2048)

#     myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')  # 获取字体，为设置中文显示
#
#     for epoch in range(10):
#             data1 = [1, 1.3, 1.39, 1.41, 1.42, 1.40,1.5,1.8,1.9,2.]
#             x= [i+20*epoch for i in range(10)]
#             plt.plot(x,data1,'k-o',label='loss')
#     plt.xlabel(u"iter", fontproperties=myfont)
#     plt.ylabel(u"loss", fontproperties=myfont)
#     plt.title("loss_curve")
#     plt.legend(['loss'])  #
#     plt.show()



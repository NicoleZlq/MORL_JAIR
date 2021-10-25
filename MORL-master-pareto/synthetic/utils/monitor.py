import visdom
import torch
import numpy as np


class Monitor(object):

    def __init__(self, train=False, spec=''):
        self.vis = visdom.Visdom(env='model1')
        self.train = train
        self.spec = spec
        self.show_num = 0
        if self.train:
            self.loss_window = self.vis.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1)).cpu(),
                opts=dict(xlabel='episode',
                          ylabel='mle loss',
                          title='Training Loss' + spec,
                          legend=['Loss']))

        self.value_window = None
        self.text_window = None
        self.time_window = None
        #**********************
        self.Q_window = None

    def update(self, eps, tot_reward, Act_1, Act_2, loss=None):
        if self.train:
            self.vis.line(
                X=torch.Tensor([eps]).cpu(),
                Y=torch.Tensor([loss]).cpu(),
                win=self.loss_window,
                update='append')

        if self.value_window == None:
            self.value_window = self.vis.line(X=torch.Tensor([eps]).cpu(),
                                              Y=torch.Tensor([tot_reward,Act_1,Act_2]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='episode',
                                                        ylabel='scalarized Q value',
                                                        title='Value Dynamics' + self.spec,
                                                        legend=['Total Reward','Act_1','Act_2']))
        else:
            self.vis.line(
                X=torch.Tensor([eps]).cpu(),
                Y=torch.Tensor([tot_reward,Act_1,Act_2]).unsqueeze(0).cpu(),
                win=self.value_window,
                update='append')

    def text(self, tt):
        if self.text_window == None:
            self.text_window = self.vis.text("QPath" + self.spec)
        self.vis.text(
            tt,
            win=self.text_window,
            append=True)

    def init_log(self, save_path, name):

        self.log_file = open("{}{}.log".format(save_path, name), 'w')

    def add_log(self, state, action, reward, terminal, preference):
        self.log_file.write("{}\t{}\t{}\t{}\t{}\n".format(state, action, reward, terminal, preference))

 #****************************************
    def show_sample_Q(self, sample):

        if self.Q_window == None:
            self.Q_window = self.vis.scatter(X=torch.zeros(1, 2),
                                             opts=dict(xlabel='Objective 1',
                                                       ylabel='Objective 2',
                                                       title ='samples q' + self.spec,
                                                       legend=['original']))
        else:
            self.show_num += 1
            self.vis.scatter(
                X=sample,
                win=self.Q_window,
                name = 'sample Q' + str(self.show_num),
                update='append')
            # for i in range(j):
            #     c = torch.tensor([list(Q[i])])
            #     print(c)
            #     print(torch.zeros((1,2)))
            #     self.vis.scatter(
            #         X = torch.tensor([list(Q[i])]),
            #         win=self.Q_window,
            #         update='append')


    def calculate_time (self, train, epoch, eps):

        if self.time_window == None:
            self.time_window = self.vis.line(X=torch.Tensor([eps]).cpu(),
                                             Y=torch.Tensor([train,epoch]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='episode',
                                                        ylabel='time',
                                                        title='train time' + self.spec,
                                                        legend=['train_time','epoch_time']))
        else:
            if train != 0:
                self.vis.line(
                    X=torch.Tensor([eps]).cpu(),
                    Y=torch.Tensor([train,epoch]).unsqueeze(0).cpu(),
                    win=self.time_window,
                    update='append')





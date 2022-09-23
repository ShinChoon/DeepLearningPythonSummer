from turtle import width
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()


class Draw_IMC(object):
    def __init__(self, total_channels=[], input_sizes=[], MLP_ports=[]):
        self.NumDots = 4
        self.NumConvMax = 8
        self.NumFcMax = 20
        self.White = 1.
        self.Light = 0.7
        self.Medium = 0.5
        self.Dark = 0.3
        Darker = 0.15
        self.Black = 0.
        self.size_list = input_sizes
        self.input_channels = total_channels[:-1]
        self.output_channels = total_channels[1:]
        self.MLP_ports = MLP_ports
        self.usage_ratio = []


        self.fc_unit_size = 1
        self.layer_width = 40
        self.patches = []
        self.colors = []
        print("len input channels: ", self.input_channels)

    def add_layer(self, size=(24, 24), num=5,
                  num_max = 8,
                  num_dots = 4,
                  top_left=[0, 0],
                  loc_diff=[3, -3],
                  ):
        # add a rectangle
        top_left = np.array(top_left)
        loc_diff = np.array(loc_diff)
        loc_start = top_left - np.array([0, size[0]])
        this_num = min(num, num_max)
        start_omit = (this_num - num_dots) // 2
        end_omit = this_num - start_omit
        start_omit -= 1

        for ind in range(this_num):
            if (num > 8) and (start_omit < ind < end_omit):
                omit = True
            else:
                omit = False

            if omit:
                self.patches.append(
                    Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
            else:
                self.patches.append(Rectangle(loc_start + ind * loc_diff,
                                              size[1], size[0]))

            if omit:
                self.colors.append('g')
            if ind % 2:
                self.colors.append('r')
            else:
                self.colors.append('y')
                
    def add_mapping(self, patch_size, position, is_conv=False):

        start_loc = position \
            # + (self.num_show_list[ind_bgn] - 1) * np.array(self.loc_diff_list[ind_bgn]) \
        _width = patch_size[1]
        _length = patch_size[0]
        if is_conv:
            _length /= 2 if _length >= 36 else 1
            _patches = [Rectangle([start_loc[0]+ind, start_loc[1]], 1, -_length)
                        for ind in range(_width)]
        else:
            _patches = [Rectangle(start_loc, _width, -_length)]

        for ind in range(len(_patches)):
            _patches[ind].set_color(
                (1 * ind/len(_patches), ind/len(_patches)**2, 1-1*ind/len(_patches)))
        self.patches.extend(_patches)
        # self.colors.append('1')
        # self.colors.append('1')
        # self.colors.append('1')
        # self.colors.append('1')
        # self.colors.append('1')
        # self.colors.append('1')
        # self.colors.append('1')

    def label(self, xy, text, xy_off=[0, 4]):
        plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
                 family='sans-serif', size=8)

    def draw_Convnet(self):

        ############################
        # conv layers
        self.x_diff_list = [0, self.layer_width, self.layer_width,
                            0, 0, self.layer_width]
        text_list = ['Inputs'] + ['Feature\nmaps'] * (len(self.size_list) - 1)
        self.loc_diff_list = [[3, -3]] * len(self.size_list)
        numer_cycle = [d-3+1 for d in self.size_list]
        # _num_show_list = [d*4 for d in self.input_channels]
        self.num_show_list = list(
            map(min, self.input_channels, [self.NumConvMax] * len(self.input_channels)))
        self.top_left_list = np.c_[
            np.cumsum(self.x_diff_list), np.zeros(len(self.x_diff_list))]
        print("self.num_show_list ", self.num_show_list)


        for ind in range(len(self.size_list)-1, -1, -1):
            self.add_layer(size=(36, 32),
                           num=1,
                           num_max=self.NumConvMax,
                           num_dots=self.NumDots,
                           top_left=self.top_left_list[ind], loc_diff=self.loc_diff_list[ind])
            self.label(self.top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
                self.input_channels[ind], self.size_list[ind], self.size_list[ind]))

        self.draw_weights(numer_cycle)

    def draw_weights(self, numer_cycle):
        # ###########################
        # in between layers
        patch_size_list = []
        for ind in range(len(self.size_list)):
            patch_size_list.append((9*self.input_channels[ind], self.output_channels[ind]))
        text_list = ['Convolution' for d in self.size_list]

        for ind in range(len(patch_size_list)):
            self.add_mapping(
                patch_size_list[ind], (self.top_left_list[ind][0],self.top_left_list[ind][1]), is_conv=True)

        for ind in range(len(patch_size_list)):
            for _rep in range(int(32/patch_size_list[ind][1])-1):
                inter_ratio = int(patch_size_list[ind][1])/4
                self.add_mapping(
                    patch_size_list[ind], (self.top_left_list[ind][0]+4*(_rep+1)*inter_ratio, self.top_left_list[ind][1]-3*(_rep+1)*inter_ratio), True)
                
            repeat_times = int(32/patch_size_list[ind][1])

            occupation = repeat_times * patch_size_list[ind][0] * patch_size_list[ind][1]/(36*32)
            fold_factor = 1
            if patch_size_list[ind][0] >= 36:
                occupation /=2
                fold_factor *= 2

            self.label(self.top_left_list[ind], text_list[ind] + '\n{}x{}x{}->{:.2%}'.format(
                int(patch_size_list[ind][0]/fold_factor), patch_size_list[ind][1], repeat_times, occupation) + '\nCycle: \n{0}x{0}/{1}*{2}={3}'.format(
                    numer_cycle[ind], repeat_times, fold_factor, numer_cycle[ind]*numer_cycle[ind]/repeat_times*fold_factor), xy_off=[6, -65]
            )
            
    def draw_weights_fl(self, num_list):
        # ###########################
        # in between layers
        patch_size_list = []
        for i in range(len(num_list)):
            patch_size_list.append((int(self.MLP_ports[i][0]/num_list[i]),self.MLP_ports[i][1]))

        text_list = ['MLP'for d in self.MLP_ports]
        for ind in range(len(self.size_list)):
            self.add_mapping(
                patch_size_list[ind], (self.top_left_list[ind][0], self.top_left_list[ind][1]), False)
            occupation = patch_size_list[ind][1] * patch_size_list[ind][0]/(36*32)
            self.label(self.top_left_list[ind], text_list[ind] + '\n{}x{}->{:.2%}'.format(
                patch_size_list[ind][0], 
                patch_size_list[ind][1], 
                occupation) + '\n\n{0}x{1}={2}'.format(
                    self.MLP_ports[ind][0], self.MLP_ports[ind][1], 
                    self.MLP_ports[ind][0]*self.MLP_ports[ind][1]), xy_off=[6, -65]
            )

    def draw_Fullconnect(self):
        ############################
        # fully connected layers
        self.size_list = [(self.fc_unit_size*36, self.fc_unit_size*32)] * len(self.MLP_ports)
        num_list = []
        for d in self.MLP_ports:
            if d[0]*d[1] % (36*32) == 0:
                num_list.append(int(d[0]*d[1]/(36*32)))
            else:
                num_list.append(1)

        self.num_show_list = list(
            map(min, num_list, [self.NumFcMax] * len(num_list)))
        self.x_diff_list = [
            sum(self.x_diff_list) + self.layer_width, self.layer_width, self.layer_width]
        self.top_left_list = np.c_[
            np.cumsum(self.x_diff_list), np.zeros(len(self.x_diff_list))]
        self.loc_diff_list = [[self.fc_unit_size, -
                               self.fc_unit_size]] * len(self.top_left_list)
        text_list = ['Hidden\nunits'] * (len(self.size_list) - 1) + ['Outputs']

        for ind in range(len(self.size_list)):
            self.add_layer(size=self.size_list[ind],
                           num=self.num_show_list[ind],
                           top_left=self.top_left_list[ind],
                           loc_diff=self.loc_diff_list[ind])
            self.label(self.top_left_list[ind], text_list[ind] + '\n{}'.format(
                num_list[ind]))

        self.draw_weights_fl(num_list)

    def draw_picture(self):
        fig, ax = plt.subplots()

        self.draw_Convnet()
        # self.draw_weights()
        self.draw_Fullconnect()
        # self.draw_weights_fl()
        ############################
        # for patch, color in zip(self.patches, self.colors):
        #     patch.set_color(color)
        #     if isinstance(patch, Line2D):
        #         ax.add_line(patch)
        #     else:
        #         patch.set_edgecolor(self.Black * np.ones(3))
        #         ax.add_patch(patch)

        for patch in self.patches:
            if isinstance(patch, Line2D):
                ax.add_line(patch)
            else:
                patch.set_edgecolor(self.Black * np.ones(3))
                ax.add_patch(patch)

        plt.tight_layout()
        plt.axis('equal')
        plt.axis('off')
        plt.show()
        fig.set_size_inches(8, 5)

        fig_dir = './'
        fig_ext = '.png'
        fig.savefig(os.path.join(fig_dir, 'IMC_fig' + fig_ext),
                    bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    model = Draw_IMC()
    model.draw_picture()

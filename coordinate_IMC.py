from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcdefaults()
fig, ax = plt.subplots()
# Major ticks every 20, minor ticks every 5
major_x_ticks = np.arange(0, 32, 4)
minor_x_ticks = np.arange(0, 32, 1)
major_y_ticks = np.arange(0, 36, 4)
minor_y_ticks = np.arange(0, 36, 1)

image_data = []

x_axis = np.arange(0, 32, 1)
y_axis = np.arange(0, 36, 1)


image_data = np.array([(i, h)for h in range(30) for i in range(30)])

class Conv_image(object):
    def __init__(self, image_data=[],  inch_num=0, outch_num=0, conv_scale=3):
        self.inch_num = inch_num
        self.out_num = outch_num
        self.image_data = np.repeat(image_data,self.inch_num)
        self.conv_scale = conv_scale
        self.xlim = 32
        self.ylim = 36

    def kernel_shape(self, start):
        """
        this function is to output a kernel array with convolution scale, start from a
        specific location
        :param start the location of the first element in a kernel
        return array of kernel
        """
        return np.array([(start[0]+h, start[1]+i)for h in range(self.conv_scale) for i in range(self.conv_scale)])

    def initialize_image(self):
        """
        return coordinates with 36x32 and ticks, ax objects
        """
        # self.image_data = np.reshape(self.image_data, (36, 32, 2))
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 36)

        ax.set_xticks(major_x_ticks)
        ax.set_xticks(minor_x_ticks, minor=True)
        ax.set_yticks(major_y_ticks)
        ax.set_yticks(minor_y_ticks, minor=True)
        ax.grid(which='both')

        return self.image_data, ax

    def decide_scan_num(self):
        """
        calcualte how many times a scanning should be 
        """
        column_num = int((self.ylim/(self.inch_num*self.conv_scale)-self.conv_scale))+1
        row_num = int(self.xlim/self.out_num)
        return min(column_num, row_num)

    def draw_weights(self, xmin, xwidth, ymin, yheight, drift, kernel=[], colors=[0, 0, 0]):
        """

        :param loc: location of one 1*9 weights
        :param drift: it is for drift in y_aix, the x_axis drift is decided already in loc
        """
        h_drift = int(drift)
        ax.broken_barh([(xmin[0], xwidth), (xmin[1], 0)],
                       (ymin, yheight), facecolors=colors)



    def draw_image(self):
        # And a corresponding grid
        self.image_data, ax = self.initialize_image()
        scan_num = self.decide_scan_num()
        locations = image_data[0:scan_num]
        for h in range(len(locations)):
            kernel =self.kernel_shape(locations[h])
            kernel = np.repeat(kernel, 2, axis=0)
            for i in range(int(self.out_num)):
                each_loc = kernel[0]*self.out_num
                each_loc[0] +=i
                drift = kernel[0][0]*self.inch_num
                self.draw_weights(xmin=each_loc,
                                    xwidth=1,
                                    ymin=each_loc[1]+drift*self.conv_scale,
                                    yheight=self.inch_num * self.conv_scale**2,
                                    drift=kernel[0][0]*self.inch_num,
                                    kernel=kernel,
                                    colors=[(1 * i/self.out_num, i/self.out_num**2, 1-1*i/self.out_num)])
                
                for i in range(2*self.conv_scale**2):
                    # for r in range(self.inch_num):
                    ax.text(each_loc[0], each_loc[1]+drift*self.conv_scale + i,
                            '{}_{}'.format(kernel[i][0], kernel[i][1]), fontsize='xx-small') 


    def save_image(self, name=""):
        fig.set_size_inches(8, 8)   
        fig_dir = './'
        fig_ext = '_{}_{}.png'.format(self.inch_num, self.out_num)
        fig.savefig(os.path.join(fig_dir, name + fig_ext),
                    bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    Conv1 = Conv_image(image_data=image_data, inch_num=2, outch_num=8, conv_scale=3)
    Conv1.draw_image()
    Conv1.save_image("coordinate_IMC")


from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcdefaults()
fig, ax = plt.subplots()


image_shape = []

x_axis = np.arange(0, 32, 1)
y_axis = np.arange(0, 36, 1)


image_shape = np.array([(i, h)for h in range(30) for i in range(30)])

class Conv_image(object):
    def __init__(self, _fig, _ax, image_shape=[],  inch_num=0, outch_num=0, conv_scale=3, window_drift = 0):
        """
        Conv layer weight mapping
        params fig: fig object from matplotlib
        params ax: axis object from matplotlib
        params image_shape: array of indexs with size initalized as image input
        params inch_num: input channel number
        params outch_num: output channel number
        params conv_scale: scale of convolution kernel
        """
        self._fig = _fig
        self._ax = _ax
        self.inch_num = inch_num
        self.out_num = outch_num
        self.image_shape = np.repeat(image_shape,self.inch_num)
        self.conv_scale = conv_scale
        self.window_drift = window_drift
        self.xlim = 32 * 1
        self.ylim = 36 * 1
        self.output_scale = self.image_shape - self.conv_scale + 1

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


        # Major ticks every 20, minor ticks every 5
        major_x_ticks = np.arange(self.xlim* self.window_drift, (1+self.window_drift)*self.xlim, 4)
        minor_x_ticks = np.arange(self.xlim* self.window_drift, (1+self.window_drift)*self.xlim, 1)

        major_y_ticks = np.arange((self.ylim-(self.ylim-self.xlim) * self.conv_scale)*self.window_drift,
                                  (1+self.window_drift)*self.ylim - (self.ylim-self.xlim) * self.conv_scale*self.window_drift, 4)
        minor_y_ticks = np.arange((self.ylim-(self.ylim-self.xlim) * self.conv_scale)*self.window_drift,
                                  (1+self.window_drift)*self.ylim - (self.ylim-self.xlim) * self.conv_scale*self.window_drift, 1)

        self._ax.set_xlim([self.window_drift*self.xlim,
                          self.xlim*(1 + self.window_drift)])

        self._ax.set_ylim([(self.ylim-(self.ylim-self.xlim) * self.conv_scale)*self.window_drift,
                          (1+self.window_drift)*self.ylim-(self.ylim-self.xlim) * self.conv_scale*self.window_drift])
        self._ax.set_xticks(major_x_ticks)
        self._ax.set_xticks(minor_x_ticks, minor=True)
        self._ax.set_yticks(major_y_ticks)
        self._ax.set_yticks(minor_y_ticks, minor=True)
        self._ax.grid(which='both')

    def decide_scan_num(self):
        """
        calcualte how many times a scanning should be 
        """
        column_num = int((self.ylim/(self.inch_num*self.conv_scale)-self.conv_scale))+1
        row_num = int(self.xlim/self.out_num)
        return min(column_num, row_num)

    def draw_weights(self, xmin, xwidth, ymin, yheight, colors=[0, 0, 0]):
        """
        :param loc: location of one 1*9 weights
        :param xmin ymin: start loction of x and y
        :param xwidth, yheight: extend length on x and y
        :param color: normalized value of RGB
        """
        self._ax.broken_barh([(xmin[0], xwidth)],
                       (ymin, yheight), facecolors=colors)



    def draw_image(self):
        """
        main process to draw the image
        return None
        """
        # And a corresponding grid
        #modify ax and fig
        self.initialize_image()
        scan_num = self.decide_scan_num()   #scan_num is how many times a bundle of mapping can be mapped
        # point to the upper left corner of kernel
        locations = image_shape[scan_num * self.window_drift\
            :scan_num*(1+self.window_drift)]
        # print("kernel[0] ", self.kernel_shape(locations[0]))
        # print("kernel[len(locations)-1] ", self.kernel_shape(locations[len(locations)-1]))
        for h in range(len(locations)):  # for each scanning time, map a bundle of kernels
            kernel =  self.kernel_shape(locations[h])   #create 1*n**2 array of kenel mapping
            kernel = np.repeat(kernel, self.inch_num, axis=0)   # repeat the element by input channels
            for i in range(int(self.out_num)):  # by output channels along x direction
                each_loc = (kernel[0])*self.out_num   # decide the start location of mapping 1*n**2 without drift in y direction
                each_loc[0] = self.window_drift*32 + h*self.out_num
                each_loc[0] +=i # drift the location along x axis by the output channels
                drift = (self.window_drift*scan_num+h)*self.inch_num * self.conv_scale  # drift along y axis,
                print("drift", drift)
                # print("kernel[0]: ", kernel[0])
                # print("each_loc; ", each_loc)
                self.draw_weights(xmin=each_loc,
                                    xwidth=1,
                                    ymin=drift,
                                    yheight=self.inch_num * self.conv_scale**2,
                                    colors=[(1 * i/self.out_num, i/self.out_num**2, 1-1*i/self.out_num)])
                
                for i in range(self.inch_num*self.conv_scale**2):
                    self._ax.text(each_loc[0], drift + i,
                            '{}_{}'.format(kernel[i][0], kernel[i][1]), fontsize='xx-small') 
                
                # print("kernel[i][0]: ", kernel[i])


    def save_image(self, name=""):
        """
        save image
        :param name: name of string
        """
        self._fig.set_size_inches(8, 8)   
        fig_dir = './'
        fig_ext = '_{}_{}.png'.format(self.inch_num, self.out_num)
        self._fig.savefig(os.path.join(fig_dir, name + fig_ext),
                    bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    Conv1 = Conv_image(_fig=fig, _ax=ax, image_shape=image_shape,
                       inch_num=2, outch_num=8, conv_scale=3, window_drift=3)
    Conv1.draw_image()
    Conv1.save_image("coordinate_IMC")


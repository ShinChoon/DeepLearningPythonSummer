from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcdefaults()
fig, ax = plt.subplots()


image_shape = []

x_axis = np.arange(0, 32, 1)
y_axis = np.arange(0, 36, 1)


class Fully_image(object):
    def __init__(self, _fig, _ax, image_shape=[30,30],  inch_num=0, outch_num=0, window_drift=0):
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
        self.image_array = np.array([(i, h)for h in range(
            image_shape[0]) for i in range(image_shape[1])])
        self.image_shape = np.repeat(
            self.image_array, self.inch_num)
        self.window_drift = window_drift
        self.xlim = 32 * 1
        self.ylim = 36 * 1

    def initialize_image(self):
        """
        return coordinates with 36x32 and ticks, ax objects
        """

        # Major ticks every 20, minor ticks every 5
        major_x_ticks = np.arange(
            self.xlim * self.window_drift, (1+self.window_drift)*self.xlim, 4)
        minor_x_ticks = np.arange(
            self.xlim * self.window_drift, (1+self.window_drift)*self.xlim, 1)

        major_y_ticks = np.arange((self.ylim-(self.ylim-self.xlim) * 1)*self.window_drift,
                                  (1+self.window_drift)*self.ylim - (self.ylim-self.xlim) * 1*self.window_drift, 4)
        minor_y_ticks = np.arange((self.ylim-(self.ylim-self.xlim) * 1)*self.window_drift,
                                  (1+self.window_drift)*self.ylim - (self.ylim-self.xlim) * 1*self.window_drift, 1)

        self._ax.set_xlim([self.window_drift*self.xlim,
                          self.xlim*(1 + self.window_drift)])

        self._ax.set_ylim([(self.ylim-(self.ylim-self.xlim) * 1)*self.window_drift,
                          (1+self.window_drift)*self.ylim-(self.ylim-self.xlim) * 1*self.window_drift])
        self._ax.set_xticks(major_x_ticks)
        self._ax.set_xticks(minor_x_ticks, minor=True)
        self._ax.set_yticks(major_y_ticks)
        self._ax.set_yticks(minor_y_ticks, minor=True)
        self._ax.grid(which='both')

    def decide_scan_num(self):
        """
        calcualte how many times a scanning should be 
        """
        column_num = int(
            (self.ylim/(self.inch_num*1)-1))+1
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
        # scan_num is how many times a bundle of mapping can be mapped
        scan_num = self.decide_scan_num()
        # point to the upper left corner of kernel
        locations = self.image_array[scan_num *
                                self.window_drift:scan_num*(1+self.window_drift)]

        for i in range(self.xlim):
            for h in range(self.ylim):
                self.draw_weights(xmin=[0+self.window_drift*self.xlim,0],
                                  xwidth=i+1,
                                  ymin=0 + self.window_drift*self.xlim,
                                  yheight=h+1,
                                  colors=[(1 * i/self.inch_num, i/self.inch_num**2, 1-1*i/self.inch_num)])
                self._ax.text(i+self.xlim*self.window_drift, h + (self.xlim)*self.window_drift,
                          '{}_{}'.format(i+self.xlim*self.window_drift, h + self.ylim*self.window_drift), fontsize='xx-small')

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
    Fully1 = Fully_image(_fig=fig, _ax=ax, image_shape=[30,30],
                       inch_num=288, outch_num=32, window_drift=0)
    Fully1.draw_image()
    Fully1.save_image("FullyIMC")

import numpy as np
import os
import ntpath
import time
from . import util as util
from . import html
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO 

class Visualizer():
    """
    A class for visualizing and saving images during training.

    Args:
        opt (object): An object containing options for visualization.

    Attributes:
        tf_log (bool): Whether to log images using TensorFlow.
        use_html (bool): Whether to save images to an HTML file.
        win_size (int): The size of the display window.
        name (str): The name of the experiment.
        tf (module): The TensorFlow module.
        log_dir (str): The directory for saving TensorFlow logs.
        writer (SummaryWriter): The TensorFlow summary writer.
        web_dir (str): The directory for saving HTML files.
        img_dir (str): The directory for saving images.
        log_name (str): The path to the loss log file.

    Methods:
        display_current_results: Displays and saves the current results.
        plot_current_errors: Plots and logs the current errors.
        print_current_errors: Prints the current errors.
        results_plot: Plots and saves the qualitative virtual stain results.
        save_images: Saves images to the disk.
    """
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = False #opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, step):
        """
        Displays and saves the current results.

        Args:
            visuals (dict): A dictionary of images to display or save.
            epoch (int): The current epoch.
            step (int): The current step.

        Returns:
            None
        """
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format='tiff')
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, i))
                        # util.save_image(image_numpy[i], img_path)
                        im = Image.fromarray(image_numpy[i])
                        im.save(img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    im = Image.fromarray(image_numpy)
                    im.save(img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.png' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    def plot_current_errors(self, errors, step):
        """
        Plots and logs the current errors.

        Args:
            errors (dict): A dictionary of error labels and values.
            step (int): The current step.

        Returns:
            None
        """
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    def print_current_errors(self, epoch, i, errors, t):
        """
        Prints the current errors.

        Args:
            epoch (int): The current epoch.
            i (int): The current iteration.
            errors (dict): A dictionary of error labels and values.
            t (float): The time taken for the current iteration.

        Returns:
            None
        """
        
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % message)

    def results_plot(self,input_x,target,predictions,titles,writer,epoch,rows):
        """
        Plots and saves the qualitative virtual stain results.

        Args:
            input_x (ndarray): The input images.
            target (ndarray): The target images.
            predictions (ndarray): The predicted images.
            titles (list): The titles for each column in the plot.
            writer (SummaryWriter): The TensorFlow summary writer.
            epoch (int): The current epoch.
            rows (int): The number of rows in the plot.

        Returns:
            None
        """

        fig, axs = plt.subplots(input_x.shape[0], 3, figsize=(10, 30))

        # Set the titles for each column
        axs[0, 0].set_title(titles[0])
        axs[0, 1].set_title(titles[1])
        axs[0, 2].set_title(titles[2])

        # Iterate over each row and plot the corresponding images
        for row in range(rows):
            # Plot the Brightfield image in the first column
            axs[row, 0].imshow(input_x[row, 0],cmap='gray')
            axs[row, 0].axis('off')

            # Plot the Fluorescence Stain image in the second column
            axs[row, 1].imshow(target[row, 0],cmap='gray')
            axs[row, 1].axis('off')

            # Plot the Virtual Stain image in the third column
            axs[row, 2].imshow(predictions[row, 0],cmap='gray') #, vmin=np.percentile(B_f[row, 0],0.05), vmax = np.percentile(B_f[row, 0],0.95))
            axs[row, 2].axis('off')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Add the plot to TensorBoard
        writer.add_figure("Qualitative Virtual Stain Results", fig,global_step=epoch)

    def save_images(self, webpage, visuals, image_path):
        """
        Saves images to the disk.

        Args:
            webpage (HTML): The HTML object for saving images.
            visuals (dict): A dictionary of images to save.
            image_path (str): The path to the image.

        Returns:
            None
        """
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

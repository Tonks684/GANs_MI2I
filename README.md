## Introduction to Generative Modelling
In this part of the exercise, we will tackle the same supervised image-to-image translation task but use an alternative approach. Here we will explore a generative modelling approach, specifically a conditional Generative Adversarial Network (cGAN). <br>

The previous regression-based method learns a deterministic mapping from phase contrast to fluorescence. This results in a single virtual staining prediction to the image translation task which often leads to blurry results. Virtual staining is an ill-posed problem; given the phase contrast image, with inherent noise and lack of contrast between the background and the structure of interest, it can be very challenging to virtually stain from the phase contrast image alone. In fact, there is a distribution of possible virtual staining solutions that could come from the phase contrast.

cGANs learn to map from the phase contrast domain to a distirbution of virtual staining solutions. This distribution can then be sampled from to produce virtual staining predictions that are no longer a compromise between possible solutions which can lead to improved sharpness and realism in the generated images. Despite these improvements, cGANs can be prown to 'hallucinations' in which the network instead of making a compromise when it does not know something (such as a fine-grain detail of the nuclei shape) it makes something up that looks very sharp and realistic. These hallucinations can appear very plausible, but in many cases to predict such details from the phase contrast is extremely challenging. This is why determining reliable evaluation criteria for the task at hand is very important when dealing with cGANs .<br>
<br>
<br>
![Overview of cGAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/GAN.jpg?raw=true)
<br>
<br>

At a high-level a cGAN has two networks; a generator and a discriminator. The generator is a fully convolutional network that takes the source image as input and outputs the target image. The discriminator is also a fully convolutional network that takes as input the source image concatentated with a real or fake image and outputs the probabilities of whether the real fluorescence image is real or whether the fake virtual stain image is fake as shown in the figure above.<br>

The generator is trained to fool the discriminator into predicting a high probability that its generated outputs are real, and the discriminator is trained to distinguish between real and fake images. Both networks are trained using an adversarial loss in a min-max game, where the generator tries to minimize the probability of the discriminator correctly classifying its outputs as fake, and the discriminator tries to maximize this probability. It is typically trained until the discriminator can no longer determine whether or not the generated images are real or fake better than a random guess (p(0.5)).<br>

We will be exploring [Pix2PixHD GAN](https://arxiv.org/abs/1711.11585) architecture, a high-resolution extension of a traditional cGAN adapted for our recent [virtual staining works](https://ieeexplore.ieee.org/abstract/document/10230501?casa_token=NEyrUDqvFfIAAAAA:tklGisf9BEKWVjoZ6pgryKvLbF6JyurOu5Jrgoia1QQLpAMdCSlP9gMa02f3w37PvVjdiWCvFhA). Pix2PixHD GAN improves upon the traditional cGAN by using a coarse-to-fine generator, a multi-scale discrimator and additional loss terms. The "coarse-to-fine" generator is composed of two sub-networks, both ResNet architectures that operate at different scales. As shown below the first sub-network (G1) generates a low-resolution image, which is then upsampled and concatenated with the source image to produce a higher resolution image. The multi-scale discriminator is composed of 3 networks that operate at different scales, each network is trained to distinguish between real and fake images at that scale using the same convolution kernel size. This leads to the convolution having a much wider field of view when the inputs are downsampled. The generator is trained to fool the discriminator at each scale. 
<br>
<br>
![Pix2PixGAN ](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_1.jpg?raw=true)
<br>
<br>
The additional loss terms include a feature matching loss (as shown below), which encourages the generator to produce images that are perceptually similar to the real images at each scale. As shown below for each of the 3 discriminators, the network takes seperaetly both phase concatenated with virtual stain and phase concatenated with fluorescence stain as input and as they pass through the network the feature maps obtained for each ith layer are extracted. We then minimize the loss which is the mean L1 distance between the feature maps obtained across each of the 3 discriminators and each ith layer. <br>
![Feature Matching Loss Pix2PixHD GAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_2.jpg?raw=true)

All of the discriminator and generator loss terms are weighted the same.

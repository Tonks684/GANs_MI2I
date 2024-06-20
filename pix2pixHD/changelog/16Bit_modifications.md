
<br><br><br>

# 16Bit changes to Code

**Base_dataset.get_transforms**

        if normalize:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        #return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return transforms.Normalize((0.5, 0.5), (0.5, 0.5))


**Pre Processing**
**util_16.save_image**

    - imsave(image_path.replace(".png",".tiff"),image_numpy[:,:,0].astype(np.float32),imagej=True)
**vizualiser_16bit**

    - from . import util_16bit, html
    - replace all util with util_16bit

**Training: train_16bit.py**

    - import util.util_16bit as util
    - from util.visualizer_16bit import Visualizer

**Post Processing & Scaling**
**util_16bit.tensor2im**

    - im_type =np.uint16
    - image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1 / 2.0 * 65535.0. #tanh  at last layour + 1 takes -1to1 to 0to2 then /2 = 0to1 then replace *255 with *65535
    -  image_numpy = np.clip(image_numpy, 0, 65535) # replace 255 with 65535.
    - ***Given the normalization in get_transforms do I not need to reverse this x = z(0.5)+0.5
<br>
 **util_16bit.sav_image**:

    - imsave(image_path,image_numpy[:,:,0].astype(np.float32),imagej=True)

**visualizer_16bit.py**:

    -from . import util_16bit as util

**Inference: test_16bit.py**

    -import util.util_16bit as util
    -from util.visualizer_16bit import Visualizer
    - from util_16bit import html

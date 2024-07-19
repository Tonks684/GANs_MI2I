import torch

def create_model(opt):
    """
    Create a model based on the given options.

    Args:
        opt (argparse.Namespace): The options for creating the model.

    Returns:
        torch.nn.Module: The created model.

    Raises:
        ImportError: If the specified model is not found.

    """
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.phase == 'train':
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print('model [%s] was created' % (model.name()))

    if opt.phase == 'train' and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model

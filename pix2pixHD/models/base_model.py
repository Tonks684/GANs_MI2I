import os
import torch
import sys

class BaseModel(torch.nn.Module):
    """
    Base class for all models in the project.
    """
    def name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return 'BaseModel'

    def initialize(self, opt):
        """
        Initialize the model.

        Args:
            opt (argparse.Namespace): The options for the model.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        """
        Set the input data for the model.

        Args:
            input: The input data.
        """
        self.input = input

    def forward(self):
        """
        Forward pass of the model.
        """
        pass

    def test(self):
        """
        Test the model.
        """
        pass

    def get_image_paths(self):
        """
        Get the paths of the input images.

        Returns:
            dict: A dictionary containing the paths of the input images.
        """
        pass

    def optimize_parameters(self):
        """
        Optimize the parameters of the model.
        """
        pass

    def get_current_visuals(self):
        """
        Get the current visual output of the model.

        Returns:
            dict: A dictionary containing the current visual output of the model.
        """
        return self.input

    def get_current_errors(self):
        """
        Get the current errors of the model.

        Returns:
            dict: A dictionary containing the current errors of the model.
        """
        return {}

    def save(self, label):
        """
        Save the model.

        Args:
            label (str): The label for the saved model.
        """
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        """
        Save a specific network of the model.

        Args:
            network: The network to be saved.
            network_label (str): The label for the network.
            epoch_label (str): The label for the epoch.
            gpu_ids: The GPU IDs to be used.
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        """
        Load a specific network of the model.

        Args:
            network: The network to be loaded.
            network_label (str): The label for the network.
            epoch_label (str): The label for the epoch.
            save_dir (str, optional): The directory to load the network from. Defaults to ''.
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        """
        Update the learning rate of the model.
        """
        pass

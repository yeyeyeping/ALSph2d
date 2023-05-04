import torchvision.transforms as T
import torch
import random


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        tensor = tensor.detach().cpu()
        noise = torch.randn(tensor.size())
        return tensor + noise * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def augment_data(img, flip=0, n_rotation=0, flip_axis=2, rot_axis0=2, rot_axis1=3, jitter=0, blur=1, mean_gaussian=0,
                 std_gaussian=0):
    """
    We apply the given transformation (flip and rotation) on the input image
    :param flip: [0 or 1] flip applied as the initial transformation
    :param flip: [0, 1, 2, 3] number of rotations applied as the initial transformation
    :param jitter:  (same) value for amount of brightness, contrast, saturation and hue jitter.
                    The factor will be uniformly from [max(0, 1 - value), 1 + value],
                    except for hue value which will be chosen between  0.5 < =[-value, value] <= 0.5
    :param blur: (same) value of kernel size and sigma for Gaussian blur. Kernel will have shape (value, value)
                 Sigma will be chosen uniformly at random between 0.1 and that value.
    """
    if flip != 0:
        img = torch.flip(img, [flip_axis])

    if n_rotation != 0:
        img = torch.rot90(img, n_rotation, [rot_axis0, rot_axis1])

    if jitter != 0:
        transform = T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter)
        img = transform(img)

    if blur != 1:
        transform = T.GaussianBlur(kernel_size=(blur, blur), sigma=(0.1, blur))
        img = transform(img)

    if mean_gaussian != 0 or std_gaussian != 0:
        transform = AddGaussianNoise(mean_gaussian, std_gaussian)
        img = transform(img)

    return img


def random_augmentation(x, aug_dic=None, flip_axis=2, rotaxis0=2, rotaxis1=3, aug_gaussian_mean=0, aug_gaussian_std=0):
    """ We do augmentation (flip, rotation, mult(0.9 - 1.1)
    :param x: a tensor of shape (#channels, x, y) or (#channels, x, y, z)
    :param aug_dic: augmentation dictionary (if given)
    :param flip_axis: tensor axis for flipping
    :param rotaxis0: tensor first axis of rotation
    :param rotaxis1: tensor second axis of rotation
    :param type: type of input ('img' or 'target'). If 'target', no jitter or blurring will be applied
    """
    if aug_dic is None:
        # We get params for number of flips (0 or 1) and number of rotations (0 ro 3)
        flip = torch.randint(0, 2, (1,)).item() if random.random() > 0.5 else 0
        num_rot = torch.randint(0, 4, (1,)).item() if random.random() > 0.5 else 0

        # We define the same value for amount of brightness, contrast, saturation and hue jitter.
        # The factor will be uniformly from [max(0, 1 - value), 1 + value],
        # except for hue value which will be chosen between  0.5 < =[-value, value] <= 0.5
        jitter = 0.5 if random.random() > 0.5 else 0

        # We define the same value for kernel size and max sigma.
        # Sigma will be chosen uniformly at random between (0.1, value)
        blur = 3 if random.random() > 0.5 else 1

        mean_gaussian = aug_gaussian_mean if random.random() > 0.5 else 0
        std_gaussian = aug_gaussian_std if random.random() > 0.5 else 0

        aug_dic = {'flip': flip,
                   'rot': num_rot,
                   'jitter': jitter,
                   'blur': blur,
                   'mean_gaussian': mean_gaussian,
                   'std_gaussian': std_gaussian
                   }
    else:
        flip = aug_dic['flip']
        num_rot = aug_dic['rot']

        # If it is a target image, there will be no jitter and bluring transformation
        jitter = 0 if type == 'target' else aug_dic['jitter']
        blur = 1 if type == 'target' else aug_dic['blur']
        mean_gaussian = 0 if type == 'target' else aug_dic['mean_gaussian']
        std_gaussian = 0 if type == 'target' else aug_dic['std_gaussian']

    # We apply the transformations
    x_aug = augment_data(x, flip=flip, n_rotation=num_rot, flip_axis=flip_axis, rot_axis0=rotaxis0, rot_axis1=rotaxis1,
                         jitter=jitter, blur=blur, mean_gaussian=mean_gaussian, std_gaussian=std_gaussian)

    return x_aug, aug_dic


def reverse_augment_data(img, flip=0, n_rotation=0, flip_axis=2, rot_axis0=2, rot_axis1=3):
    """
    We reverse the transformation (flip and rotation) of the given image
    :param flip: [0 or 1] flip applied as the initial transformation
    :param flip: [0, 1, 2, 3] number of rotations applied as the initial transformation
    """
    if n_rotation != 0:
        img = torch.rot90(img, 4 - n_rotation, [rot_axis0, rot_axis1])

    if flip != 0:
        img = torch.flip(img, [flip_axis])

    return img


def augments_forward(img, model, output, num_augmentations, device):
    augment_output = [output]
    for _ in range(num_augmentations):
        transformed_data, aug_dict = random_augmentation(img, flip_axis=2, rotaxis0=2, rotaxis1=3)
        transformed_data = transformed_data.to(device)
        transformed_output, _ = model(transformed_data)
        rev_output = reverse_augment_data(transformed_output, aug_dict['flip'], aug_dict['rot'], flip_axis=2,
                                          rot_axis0=2, rot_axis1=3).softmax(1)
        augment_output.append(rev_output)
    return torch.stack(augment_output)

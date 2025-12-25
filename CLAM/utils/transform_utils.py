from torchvision import transforms


def get_eval_transforms(mean, std, target_img_size=-1, model=None):
    """
    Args:
        mean: mean of the dataset
        std: std of the dataset
        target_img_size: target image size
        model: model specific transforms
    """

    if model == 'gigap':
        trsforms = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        trsforms = transforms.Compose(trsforms)
    else:
        trsforms = []

        if target_img_size > 0:
            trsforms.append(transforms.Resize(target_img_size))
        trsforms.append(transforms.ToTensor())
        trsforms.append(transforms.Normalize(mean, std))
        trsforms = transforms.Compose(trsforms)

    return trsforms
from PIL import Image
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize

from capcontact.config import SUPERRES_CONFIG


def get_bicubic_image(tensor):
    transform = Compose([
                         ToPILImage(),
                         Resize((tensor.size(1)*SUPERRES_CONFIG.FACTOR, 
                                 tensor.size(2)*SUPERRES_CONFIG.FACTOR), 
                                interpolation=Image.BICUBIC),
                         ToTensor()
                        ])
    return transform(tensor)


def get_image(tensor):
    transform = Compose([
                         ToPILImage(),
                         ToTensor()
                        ])
    return transform(tensor)

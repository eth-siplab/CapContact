import torch
import torch.nn as nn
from torch import autograd


from capcontact.config import TRAINING_CONFIG


def weighted_mse_loss(true_contact, pred_contact, weight_tensor):
    return torch.sum(weight_tensor*torch.square((pred_contact - true_contact)))/true_contact.size(0)


def get_generator_loss(true_contact, pred_contact, weight_tensor, critic):
    mse_loss = weighted_mse_loss(true_contact, pred_contact, weight_tensor)
    adversarial_loss = -critic(pred_contact).mean()
    return mse_loss+adversarial_loss


def gradient_penalty(true_contact, pred_contact, critic):
    batch_size = true_contact.size()[0]

    interpolation_factor = torch.rand(batch_size, 1, 1, 1)
    interpolation_factor = interpolation_factor.expand_as(true_contact)
    interpolation_factor = interpolation_factor.to(pred_contact.device)
    
    interpolated = interpolation_factor * true_contact.data + (1 - interpolation_factor) * pred_contact.data
    interpolated = autograd.Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(pred_contact.device)

    interpolated_score = critic(interpolated)

    gradients = autograd.grad(outputs=interpolated_score, inputs=interpolated,
                              grad_outputs=torch.ones(interpolated_score.size()).to(pred_contact.device),
                              create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()


def get_critic_loss(true_contact, pred_contact, critic):
    true_contact_score = critic(true_contact)
    pred_contact_score = critic(pred_contact)
    critic_loss = pred_contact_score.mean() - true_contact_score.mean() 
    critic_loss += TRAINING_CONFIG.GP_WEIGHT*gradient_penalty(true_contact, pred_contact, critic)
    return critic_loss


def get_iou(true_contact, pred_contact, threshold_value=0.5):
    threshold_contact = nn.functional.threshold(pred_contact, threshold_value, 0)

    union = torch.logical_or(threshold_contact, true_contact)
    union_area = torch.sum(union, (1, 2, 3))

    intersect = torch.logical_and(threshold_contact, true_contact)
    intersect_area = torch.sum(intersect, (1, 2, 3))

    iou = torch.true_divide(intersect_area, union_area).float()
    return iou


def get_mse(true_contact, pred_contact, threshold_value=0.5):
    return torch.nn.functional.mse_loss(pred_contact, true_contact, reduction="none").view(pred_contact.size(0), -1).mean(1)

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq, name_list, loss_ratio
from data_gen import FaceAttributesDataset
from models import FaceAttributesModel
from utils import parse_args, save_checkpoint, AverageMeter, LossMeterBag, clip_gradient, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = FaceAttributesModel()
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    MSELoss = nn.MSELoss().to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = FaceAttributesDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_dataset = FaceAttributesDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterions=(MSELoss, CrossEntropyLoss),
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('Train_Loss', train_loss, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterions=(MSELoss, CrossEntropyLoss),
                           logger=logger)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterions, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    loss_bag = LossMeterBag(name_list)
    MSELoss, CrossEntropyLoss = criterions

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        age, pitch, roll, yaw, beauty, expression, face_prob, face_shape, face_type, gender, glasses, race = label
        age_label = age.type(torch.FloatTensor).to(device)  # [N, 1]
        pitch_label = pitch.type(torch.FloatTensor).to(device)  # [N, 1]
        roll_label = roll.type(torch.FloatTensor).to(device)  # [N, 1]
        yaw_label = yaw.type(torch.FloatTensor).to(device)  # [N, 1]
        beauty_label = beauty.type(torch.FloatTensor).to(device)  # [N, 1]
        expression_label = expression.to(device)  # [N, 1]
        face_prob_label = face_prob.type(torch.FloatTensor).to(device)  # [N, 1]
        face_shape_label = face_shape.to(device)  # [N, 1]
        face_type_label = face_type.to(device)  # [N, 1]
        gender_label = gender.to(device)  # [N, 1]
        glasses_label = glasses.to(device)  # [N, 1]
        race_label = race.to(device)  # [N, 1]
        # Forward prop.
        output = model(img)  # embedding => [N, 512]
        age_out, pitch_out, roll_out, yaw_out, beauty_out, expression_out, face_prob_out, face_shape_out, face_type_out, gender_out, glasses_out, race_out = output

        # Calculate loss
        age_loss = MSELoss(age_out, age_label) * loss_ratio
        pitch_loss = MSELoss(pitch_out, pitch_label) * loss_ratio
        roll_loss = MSELoss(roll_out, roll_label) * loss_ratio
        yaw_loss = MSELoss(yaw_out, yaw_label) * loss_ratio
        beauty_loss = MSELoss(beauty_out, beauty_label) * loss_ratio
        expression_loss = CrossEntropyLoss(expression_out, expression_label)
        face_prob_loss = MSELoss(face_prob_out, face_prob_label)
        face_shape_loss = CrossEntropyLoss(face_shape_out, face_shape_label)
        face_type_loss = CrossEntropyLoss(face_type_out, face_type_label)
        gender_loss = CrossEntropyLoss(gender_out, gender_label)
        glasses_loss = CrossEntropyLoss(glasses_out, glasses_label)
        race_loss = CrossEntropyLoss(race_out, race_label)
        loss = (age_loss + pitch_loss + roll_loss + yaw_loss + beauty_loss + face_prob_loss + expression_loss +
                face_shape_loss + face_type_loss + gender_loss + glasses_loss + race_loss) / 12

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        loss_bag.update(
            [age_loss.item(), pitch_loss.item(), roll_loss.item(), yaw_loss.item(), beauty_loss.item(),
             expression_loss.item(), face_prob_loss.item(), face_shape_loss.item(),
             face_type_loss.item(), gender_loss.item(), glasses_loss.item(), race_loss.item()])

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                        'Detail: {3}'.format(epoch, i, len(train_loader), str(loss_bag), loss=losses))

    return losses.avg


def valid(valid_loader, model, criterions, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()
    loss_bag = LossMeterBag(name_list)
    MSELoss, CrossEntropyLoss = criterions

    # Batches
    for i, (img, label) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)

        # Forward prop.
        output = model(img)  # embedding => [N, 512]

        # Calculate loss
        age, pitch, roll, yaw, beauty, expression, face_prob, face_shape, face_type, gender, glasses, race = label
        age_label = age.type(torch.FloatTensor).to(device)  # [N, 1]
        pitch_label = pitch.type(torch.FloatTensor).to(device)  # [N, 1]
        roll_label = roll.type(torch.FloatTensor).to(device)  # [N, 1]
        yaw_label = yaw.type(torch.FloatTensor).to(device)  # [N, 1]
        beauty_label = beauty.type(torch.FloatTensor).to(device)  # [N, 1]
        expression_label = expression.to(device)  # [N, 1]
        face_prob_label = face_prob.type(torch.FloatTensor).to(device)  # [N, 1]
        face_shape_label = face_shape.to(device)  # [N, 1]
        face_type_label = face_type.to(device)  # [N, 1]
        gender_label = gender.to(device)  # [N, 1]
        glasses_label = glasses.to(device)  # [N, 1]
        race_label = race.to(device)  # [N, 1]

        age_out, pitch_out, roll_out, yaw_out, beauty_out, expression_out, face_prob_out, face_shape_out, face_type_out, gender_out, glasses_out, race_out = output

        # Calculate loss
        age_loss = MSELoss(age_out, age_label) * loss_ratio
        pitch_loss = MSELoss(pitch_out, pitch_label) * loss_ratio
        roll_loss = MSELoss(roll_out, roll_label) * loss_ratio
        yaw_loss = MSELoss(yaw_out, yaw_label) * loss_ratio
        beauty_loss = MSELoss(beauty_out, beauty_label) * loss_ratio
        expression_loss = CrossEntropyLoss(expression_out, expression_label)
        face_prob_loss = MSELoss(face_prob_out, face_prob_label)
        face_shape_loss = CrossEntropyLoss(face_shape_out, face_shape_label)
        face_type_loss = CrossEntropyLoss(face_type_out, face_type_label)
        gender_loss = CrossEntropyLoss(gender_out, gender_label)
        glasses_loss = CrossEntropyLoss(glasses_out, glasses_label)
        race_loss = CrossEntropyLoss(race_out, race_label)
        loss = (age_loss + pitch_loss + roll_loss + yaw_loss + beauty_loss + face_prob_loss + expression_loss +
                face_shape_loss + face_type_loss + gender_loss + glasses_loss + race_loss) / 12

        # Keep track of metrics
        losses.update(loss.item())
        loss_bag.update(
            [age_loss.item(), pitch_loss.item(), roll_loss.item(), yaw_loss.item(), beauty_loss.item(),
             expression_loss.item(), face_prob_loss.item(), face_shape_loss.item(),
             face_type_loss.item(), gender_loss.item(), glasses_loss.item(), race_loss.item()])

    # Print status
    logger.info('Validation: Loss {0:.4f}\nDetail: {1}\n'.format(losses.avg, str(loss_bag)))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()

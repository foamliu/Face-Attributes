import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq, loss_ratio
from data_gen import FaceAttributesDataset
from models import FaceAttributeModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger


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
        model = FaceAttributeModel()
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
    L1Loss = nn.L1Loss().to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = FaceAttributesDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_dataset = FaceAttributesDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # scheduler.step(epoch)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterions=(L1Loss, CrossEntropyLoss),
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('Train_Loss', train_loss, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterions=(L1Loss, CrossEntropyLoss),
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
    L1Loss, CrossEntropyLoss = criterions

    # Batches
    for i, (img, reg, expression, gender, glasses, race) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        reg_label = reg.type(torch.FloatTensor).to(device)  # [N, 5]
        expression_label = reg.type(torch.LongTensor).to(device)  # [N, 3]
        gender_label = reg.type(torch.LongTensor).to(device)  # [N, 2]
        glasses_label = reg.type(torch.LongTensor).to(device)  # [N, 3]
        race_label = reg.type(torch.LongTensor).to(device)  # [N, 4]

        # Forward prop.
        output = model(img)  # embedding => [N, 512]
        print(output.size())
        reg_out = output[:, :5]
        expression_out = output[:, 5:8]
        gender_out = output[:, 8:10]
        glasses_out = output[:, 10:13]
        race_out = output[:, 13:17]

        # Calculate loss
        reg_loss = L1Loss(reg_out, reg_label) * loss_ratio
        expression_loss = CrossEntropyLoss(expression_out, expression_label)
        gender_loss = CrossEntropyLoss(gender_out, gender_label)
        glasses_loss = CrossEntropyLoss(glasses_out, glasses_label)
        race_loss = CrossEntropyLoss(race_out, race_label)

        loss = reg_loss + expression_loss + gender_loss + glasses_loss + race_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, criterions, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()
    L1Loss, CrossEntropyLoss = criterions

    # Batches
    for i, (img, reg, expression, gender, glasses, race) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)
        reg_label = reg.type(torch.FloatTensor).to(device)  # [N, 5]
        expression_label = reg.type(torch.LongTensor).to(device)  # [N, 3]
        gender_label = reg.type(torch.LongTensor).to(device)  # [N, 2]
        glasses_label = reg.type(torch.LongTensor).to(device)  # [N, 3]
        race_label = reg.type(torch.LongTensor).to(device)  # [N, 4]

        # Forward prop.
        output = model(img)  # embedding => [N, 512]
        reg_out = output[:, :5]
        expression_out = output[:, 5:8]
        gender_out = output[:, 8:10]
        glasses_out = output[:, 10:13]
        race_out = output[:, 13:17]

        # Calculate loss
        reg_loss = L1Loss(reg_out, reg_label) * loss_ratio
        expression_loss = CrossEntropyLoss(expression_out, expression_label)
        gender_loss = CrossEntropyLoss(gender_out, gender_label)
        glasses_loss = CrossEntropyLoss(glasses_out, glasses_label)
        race_loss = CrossEntropyLoss(race_out, race_label)

        loss = reg_loss + expression_loss + gender_loss + glasses_loss + race_loss

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('Validation: Loss {0:.4f}\n'.format(losses.avg))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()

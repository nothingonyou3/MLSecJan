import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from MLSecJan.datasets import get_dataset, DATASETS
from MLSecJan.architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from MLSecJan.train_utils import AverageMeter, accuracy, init_logfile, log
from robustbench import load_model
from robustbench.data import load_cifar10

# Manually set argument values
args = {
    "dataset": "cifar10",  # Example: 'cifar10' or 'imagenet'
    "arch": "cifar_resnet110",  # Example: 'resnet50' or 'cifar_resnet110'
    "outdir": "/path/to/output/folder",  # Change this to your desired output directory
    "workers": 4,
    "epochs": 90,
    "batch": 256,
    "lr": 0.1,
    "lr_step_size": 30,
    "gamma": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "noise_sd": 0.0,
    "gpu": None,  # You can specify a GPU id if necessary
    "print_freq": 10,
}

def main():
    if args["gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    if not os.path.exists(args["outdir"]):
        os.mkdir(args["outdir"])

    train_dataset = get_dataset(args["dataset"], "train")
    test_dataset = get_dataset(args["dataset"], "test")
    pin_memory = args["dataset"] == "imagenet"
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args["batch"],
        num_workers=args["workers"],
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args["batch"],
        num_workers=args["workers"],
        pin_memory=pin_memory,
    )

    print("Pre loading model")

    # Loading the model (assuming you need robustbench)
    model = load_model(model_name='Sehwag2021Proxy_R18', dataset='cifar10', threat_model='L2')

    print("Post - loading")

    model = get_architecture(args["arch"], args["dataset"])

    logfilename = os.path.join(args["outdir"], "log.txt")
    init_logfile(
        logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc"
    )

    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args["lr"],
        momentum=args["momentum"],
        weight_decay=args["weight_decay"],
    )
    scheduler = StepLR(optimizer, step_size=args["lr_step_size"], gamma=args["gamma"])

    print("Before training")
    for epoch in range(args["epochs"]):
        scheduler.step(epoch)
        before = time.time()
        print("time setted")
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, epoch, args["noise_sd"]
        )
        test_loss, test_acc = test(test_loader, model, criterion, args["noise_sd"])
        after = time.time()
        print("ends the testing")

        log(
            logfilename,
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch,
                str(datetime.timedelta(seconds=(after - before))),
                scheduler.get_lr()[0],
                train_loss,
                train_acc,
                test_loss,
                test_acc,
            ),
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "arch": args["arch"],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args["outdir"], "checkpoint.pth.tar"), 
        )


def train(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer: Optimizer,
    epoch: int,
    noise_sd: float,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    model.train()

    for i, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        inputs = inputs + torch.randn_like(inputs, device="cpu") * noise_sd

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["print_freq"] == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            data_time.update(time.time() - end)

            inputs = inputs + torch.randn_like(inputs, device="cpu") * noise_sd

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args["print_freq"] == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

        return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()

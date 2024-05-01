import timm


def get_network(args, class_num=10, pretrain=False):
    """ return given network
    """

    net = timm.create_model(args.net, pretrained=pretrain, num_classes=class_num)

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

import timm


def get_network(args, class_num=10, pretrain=False):
    """ return given network
    """

    model = timm.create_model(args.model, pretrained=pretrain, num_classes=class_num)

    if args.gpu: #use_gpu
        model = model.cuda()

    return model

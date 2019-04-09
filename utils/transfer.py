def freeze_module(layer, depth=None):
    if depth is None:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for weight in layer.all_weights[depth]:
            weight.requires_grad = False


def train_module(layer, depth=None):
    if depth is None:
        for param in layer.parameters():
            param.requires_grad = True
    else:
        for weight in layer.all_weights[depth]:
            weight.requires_grad = True


def dict_rename_by_pattern(from_dict, patterns):
    for k in list(from_dict.keys()):
        v = from_dict.pop(k)
        p = list(filter(lambda x: x in k, patterns.keys()))
        if len(p) > 0:
            new_key = k.replace(p[0], patterns[p[0]])
            from_dict[new_key] = v
        else:
            from_dict[k] = v


def load_state_dict_subset(model, pretrained_dict):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

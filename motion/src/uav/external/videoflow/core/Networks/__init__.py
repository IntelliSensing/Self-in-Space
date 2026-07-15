def build_network(cfg):
    name = cfg.network
    if name != "MOFNetStack":
        raise ValueError(f"Unsupported VideoFlow network: {name}")

    from .MOFNetStack.network import MOFNet as network
    return network(cfg[name])

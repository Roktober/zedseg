from gen import comb_generate, h5_generate, png_generate, cvat_generate

types = {
    'combo': comb_generate,
    'h5': h5_generate,
    'png': png_generate,
    'cvat': cvat_generate
}


def create_generator(definition, device=None):
    gen_type = definition['type']
    if gen_type == 'combo':
        items = [create_generator(item) for item in definition['items']]
        return comb_generate(items, resolution=tuple(definition['resolution']), device=device)
    return types[gen_type](**{k: v for k, v in definition.items() if k != 'type'})

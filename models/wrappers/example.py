
input_size = 784
num_classes = 10

x = Input(input_size)

discriminators = wz.stack([
    wz.concat([
        Ram(mapping=mapping)(x) for mapping in complete_mapping(input_size, n)
    ], axis=0)
for i in range(num_classes)], axis=1)

out = discriminators.sum(axis=1)




input_size = 784
num_classes = 10

x = Input(input_size)

out = []
for i in range(num_classes):
    discs.append(wz.sum([
        Ram(mapping=mapping)(x) for mapping in complete_mapping(input_size, n)
    ]))

out = wz.ndarray(out)



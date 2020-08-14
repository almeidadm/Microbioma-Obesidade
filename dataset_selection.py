
def format(data):
    data = data.split('\n')
    data = data[:-1]

    for i in range(len(data)):
        data[i] = data[i].split('\t')

    feature = []
    feature.append(data[0][0])
    feature.append(data[1][0])
    feature.append(data[4][0])
    for i in range(211,len(data)):
        feature.append(data[i][0])

    samples = []
    for i in range(len(data[0])):
        aux = []
        aux.append(data[0][i])
        aux.append(data[1][i])
        aux.append(data[4][i])
        for j in range(211, len(data)):
            aux.append(data[j][i])
        samples.append(aux)

    samples = samples[1:]

    obesity = []
    for i in range(len(samples)):
        if samples[i][0] == 'Chatelier_gut_obesity':
            obesity.append(samples[i])

    return feature, obesity


if __name__ == '__main__':
    file = open('./data/abundance.txt', 'r')
    feature, obesity = format(file.read())
    file.close()

    for i in range(len(feature)):
       print(i, feature[i], obesity[0][i])

    new_data = open('./data/obesity_abundance.txt', 'w+')

    for i in range(len(feature)):
        line = str(feature[i]) + ','
        for j in range(len(obesity)):
            line += str(obesity[j][i]) + ','
        line += '\n'
        new_data.write(line)

    new_data.close()

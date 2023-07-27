import random


def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return 'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'
    pass
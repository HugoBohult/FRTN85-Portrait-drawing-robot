from RapidGenerator import generator
from EdgeDraw import segment

if __name__ == "__main__":
    countours = segment("LagetEfterPH.png", draw=False)
    generator(countours)
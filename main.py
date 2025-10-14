from RapidGenerator import generator
from EdgeDraw import segment

if __name__ == "__main__":
    countours = segment("Kalle.png", draw=False)
    generator(countours)
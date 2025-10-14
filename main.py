from RapidGenerator import generator
from EdgeDraw import segment

if __name__ == "__main__":
    countours = segment("Edward.jpg")
    generator(countours)
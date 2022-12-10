import wget
import csv


with open("images.txt","r") as f:
    line = csv.reader(f)
    for i in line:
        wget.download(i[0])
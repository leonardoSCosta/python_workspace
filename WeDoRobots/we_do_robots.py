text = "We do robots"

for letter in text:
    bin_letter = str(bin(int.from_bytes(letter.encode(), 'big')))[2:]
    if len(bin_letter) < 8:
        bin_letter = '0'*(8 - len(bin_letter)) + bin_letter
    print("{}: {}".format(letter, bin_letter))

# make_banknote_data.py
# input: raw banknote_100.txt
# output: banknote data in CNTK two-node format to screen
# for scraping (manually divide into train/test)

fin = open(".\\banknote_100_raw.txt", "r")

for line in fin:
    line = line.strip()
    tokens = line.split(",")
    if tokens[4] == "0":
        print("|stats %12.8f %12.8f %12.8f %12.8f |forgery 0 1 |# authentic" %(float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])))
    else:
        print("|stats %12.8f %12.8f %12.8f %12.8f |forgery 1 0 |# fake" %(float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])))

fin.close()
import argparse
parser = argparse.ArgumentParser(description='Sort Files')
parser.add_argument('-file', type=str, help="file name")
args = parser.parse_args()

f = open(args.file, "r")
l = []
line = f.readline()
while line:
    line = line.strip()
    l.append(line)
    line = f.readline()
l.sort(key = len)
f.close()

wf = open("sorted_" + args.file, "w")
for i in l:
	wf.write(i + "\n")
wf.close()
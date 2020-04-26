import sys
import os
# import keras
import PIL
import numpy
import pandas
# import tensorflow

def main():
  print('Hello world!')
  print(os.getcwd())
  open('/root/shared/results/hello.txt','w').write('Hello BOINC-CRYOGEL')

if __name__ == "__main__":
  main()
  sys.exit()

import os

print ('The specific model being run is: Ours')
os.system("python ./test.py --save_folder PASCAL_eval/ --dataset PASCAL --trained_model Ours")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

# print ('The specific model being run is:   Robust')
# os.system("python ./test.py --save_folder PASCAL_eval/ --dataset PASCAL  --trained_model Robust")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')
#
# print ('The specific model being run is:   SKD')
# os.system("python ./test.py --save_folder PASCAL_eval/ --dataset PASCAL  --trained_model SKD")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')
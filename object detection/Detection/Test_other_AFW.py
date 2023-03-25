import os

print ('The specific model being run is: Ours')
os.system("python ./test.py --save_folder AFW_eval/ --dataset AFW --trained_model Ours")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

# print ('The specific model being run is:   Robust')
# os.system("python ./test.py --save_folder AFW_eval/ --dataset AFW  --trained_model Robust")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')
#
#
# print ('The specific model being run is:   SKD')
# os.system("python ./test.py --save_folder AFW_eval/ --dataset AFW  --trained_model SKD")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')
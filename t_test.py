import scipy
from scipy import stats

# isbi_vst1s = 'results_dual/m-vst1s_dsbn_t1t2_gan_fpl40+dual+fpl01_i-t2-test/test_tumor_dice_all.csv'
# isbi_vst2s = 'results_dual/cyc12_vst1s_1t1+2t2_121_t2_test/valid_tumor_dice_all.csvv'
# tmi_wjh_vst1s = 'results_dual/cyc12_vst1s_121_dsbn_wi+wp+pre_t2_test/test_tumor_dice_all.csv'
# tmi_wjh_vst2s = 'results_dual/m-cyc12_vst1s_3t1+2t2_121_i-t2-test/valid_tumor_dice_all.csv'

# isbi_vst1s = 'results_dual/cyc12_vst1s_t2+pl100_t2_test/valid_tumor_assd_all.csv'
# isbi_vst2s = '/data2/jianghao/VS/SIFA/results/mmw_mr-ct/test_tumor_dice_all.csv'
# tmi_wjh_vst1s = 'result/unet_tri_vst1s/valid_tumor_dice_all.csv'
# tmi_wjh_vst2s = '/data2/jianghao/VS/vs_seg2021/config_dual/data_mmwh/result/mr-ct_s_tmi_ct_test/test_tumor_assd_all.csv'


# isbi_vst2s = 'results_dual/m-vst1s_dsbn_t1+t1-t2-cyc-t1_i-t2-valid/tumor_dice_all.csv'
# tmi_wjh_vst2s = 'results_dual/m-cyc12_vst1s_3t1+2t2_121_i-t2-test/valid_tumor_dice_all.csv'

isbi_vst2s = 'results_dual/m-vst2s_dsbn_t2+t2-t1-cyc_i-t1-valid/tumor_dice_all.csv'
tmi_wjh_vst2s = 'results_dual/cyc12_vst2s_3t2+2t1_121_t1_test/valid_tumor_dice_all.csv'

import csv
data1 = []
data2 = []
clas = 1
with open(isbi_vst2s, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        value = float(row[clas])  # Convert the string to float
        data1.append(value)
# print(data1)
with open(tmi_wjh_vst2s, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        value = float(row[clas])  # Convert the string to float
        data2.append(value)
# print(data2)
print(stats.ttest_rel(data1, data2))
isbi_vst2s = isbi_vst2s.replace('dice','assd')
tmi_wjh_vst2s = tmi_wjh_vst2s.replace('dice','assd')

import csv
data1 = []
data2 = []
clas = 1
with open(isbi_vst2s, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        value = float(row[clas])  # Convert the string to float
        data1.append(value)
# print(data1)
with open(tmi_wjh_vst2s, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        value = float(row[clas])  # Convert the string to float
        data2.append(value)
# print(data2)
print(stats.ttest_rel(data1, data2))


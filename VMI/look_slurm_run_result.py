import os

startno = 553351
endno = 553478
slurm_arr = list(range(startno, endno+1))

id_list = [9,12,13,14,15,17,18,19,0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,2,6,8,10,12,13,14,15,16,19,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,2,4,6,7,8,10,12,13,16,18,19,0,1,2,3,5,6,7,8,9,10,11,12,13,16,18,2,4,5,6,7,8,10,12,16,17,18,19,1,2,3,4,5,6,7,8,9,10,14,15,0,1,2,3,4,5,6,7,11,14,15,16,18,19,0,1,2,3,4,5,6,7,11,12,14,18,19]
# dict of slurm:id
slurm_id = {}
if len(id_list) == 0:
    id = 0
    for num in slurm_arr:
        slurm_id[num] = id
        if id == 19:
            id = 0
        else:
            id += 1
else:
    if len(id_list) != len(slurm_arr):
        print('id number is not equal to slurm number!')
        exit()
    for i, num in enumerate(slurm_arr):
        slurm_id[num] = id_list[i]

# find the abnormal slurms
abn_slurms = []
for num in slurm_arr:
    file = f'slurm-{num}.out'
    print(f'processing {file}')
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_cout = len(lines)
        i = line_cout-1
        while i >= line_cout-10:
            if 'best' in lines[i]:
                break
            i -= 1
        if i == line_cout-11:
            abn_slurms.append(num)



# abnormal slurms
abn_ids = []
print(f'>>>abnormal slurms({len(abn_slurms)}):')
for i in range(len(abn_slurms)):
    print(abn_slurms[i], end=' ')
    abn_ids.append(slurm_id[abn_slurms[i]])
print('\n')

#print abn slurms ids
print(f'>>>abnormal ids({len(abn_ids)}):')
for id in abn_ids:
    print(id, ' ')
print('\n')






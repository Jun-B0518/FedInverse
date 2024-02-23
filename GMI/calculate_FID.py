from fid_score import calculate_fid_given_paths

dataset = 'celeba'
batchsise = 50
cuda = 1
dims = 2048
attacklabel = 'VGG16_0.000&0.000_87.63.tar-seeds10-iter2000'

fid_value = calculate_fid_given_paths(dataset,
                                      [f'../attack_dataset/{dataset}/trainset/',
                                       f'attack_res/{dataset}/{attacklabel}/all/'],
                                      batchsise, cuda, dims)
print(f'FID:{fid_value:.4f}')
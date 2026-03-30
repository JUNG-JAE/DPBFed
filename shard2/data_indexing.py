import os
import re
import parameter as p

def worker_to_index(data_type, input_migrate_shard_list):
    migrate_shard_list = {}
    for shard_id in input_migrate_shard_list:
        migrate_shard_list.update({shard_id: []})
        worker_list = input_migrate_shard_list[shard_id]
        for worker_id in worker_list:
            if worker_id != "worker0":
                data_class_list = []
                for cla in p.sub_label:
                    data_class_dict = {}
                    path = './data/' + str(worker_id) + '/' + data_type + '/' + cla + '/'
                    data_index = os.listdir(path)
                    data_index = [re.sub('[.jpg]', '', _) for _ in data_index]
                    try:
                        if ".DS_Store" in data_index:
                            data_index.remove(".DS_Store")
                    except:
                        pass
                    data_class_dict[cla] = data_index
                    data_class_list.append(data_class_dict)
                # print(worker_id, data_class_list[5])
                migrate_shard_list[shard_id].append(data_class_list)
    # print(migrate_shard_list["shard2"][0][3])
    return migrate_shard_list


# temp = worker_to_index("train", temp_list)


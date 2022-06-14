import os, os.path, time
def rename(file):
    for i in range(1,106):
        # os.chdir(file)
        # items = os.listdir
        items = os.listdir(file)
        items = [item for item in items if "crossmoda" in item]
        # print(items,"999")
        # print(os.getcwd())
        for name in items:
            # print(name)
            # if not os.path.isdir(name):
            if "_"+str(i)+"_" in name:
                new_name = name.replace("_"+str(i)+"_", "_"+str(i+300)+"_")
                os.renames(name,new_name)
            # else:
                # rename(file+"\\"+name,keyword)
                # os.chdir("...")
        print("-----------------------------------------------------------------------")
        # items = os.listdir(file)
        # for name in items:
            # print(name)
rename('/mnt/39E12DAE493BA6C1/wujianghao/data/data_nii/source_training_crop')
    
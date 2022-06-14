import os, os.path, time

import os ,os.path ,time

def rename(file,keyword):
    ''' file: 文件路径    keyWord: 需要修改的文件中所包含的关键字 '''
    # start =time.clock()
    os.chdir(file)
    items = os.listdir(file)
    print(os.getcwd())
    for name in items :
        print(name)
        # 遍历所有文件
        if not os.path.isdir(name):
            if keyword in name :
                new_name = name.replace(keyword, "_"+str(i)+"_Label")
                os.renames(name,new_name)
        else:
            rename(file + '\\' + name, keyword)
            os.chdir('...')      
    print('-----------------------分界线------------------------')
    items = os.listdir(file)
    for name in items:
        print(name)
for i in range(106,243):
    rename('/mnt/39E12DAE493BA6C1/wujianghao/0813/data/T2_TRAIN_2/seg/', "_"+str(i)+"_hrT2")

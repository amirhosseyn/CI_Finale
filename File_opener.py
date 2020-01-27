import os,csv,numpy

addresses={}
addresses['1']='csv\\2clstrain1200.csv'
addresses['2']='csv\\4clstrain1200.csv'
addresses['3']='csv\\5clstrain1500.csv'


def read(file_number):
    data={}
    address=addresses[str(file_number)]
    id=0
    with open(os.path.dirname(os.path.realpath(__file__)) + address, newline='', encoding='utf-8') as dataset:
        reader = csv.reader(dataset)
        for row in reader:
            if(len(row)>0):
                d = {}
                d["x"] = row[0]
                d["y"] = row[1]
                d["class"] = row[2]
                data[id]=d
                id+=1
    new_data=numpy.zeros([len(data),3])
    for i in range(len(data)):
        new_data[i][0]=data[i]["x"]
        new_data[i][1]=data[i]["y"]
        new_data[i][2]=data[i]["class"]
    numpy.random.shuffle(new_data)
    return new_data

def create_learn(data_set):
    file_number=data_set
    data=read(file_number)
    # print(data)
    # print(len(data))
    learnlength=int(0.7*len(data))
    testlength=int(0.3*len(data))
    # print(data[0])
    # print(data[learnlength])
    numpy.savetxt('data\\learn'+ str(file_number) +'.txt',data[:learnlength],fmt='%3.9f')
    numpy.savetxt('data\\test'+ str(file_number) +'.txt',data[-testlength:],fmt='%3.9f')

def init_files():
    for i in range(len(addresses)):
        create_learn(i)
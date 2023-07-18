import torch 

def convert_spare_to_dense_connection(dict_connection):
    #convert dic_connection to spare connection
    rows , cols = [] , [] 
    for word_id in dict_connection.keys():
        list_sent_id = dict_connection[word_id] # list sent id, which consist of this word
        for sent_id in list_sent_id:
            rows.append(word_id)
            cols.append(sent_id) 
    indexes = [ rows , cols ]
    values = [1] * len(indexes[0]) 
    s = torch.sparse_coo_tensor(indexes, values )
    print(s.to_dense())

dict_connection = { 0:[10 ,11] , 1:[2 ,3 ] }
convert_spare_to_dense_connection(dict_connection)

import re
import torch

class fp_search():
    def __init__(self):
        self.path = './file/search_smile.txt'
        
    def replacex(self,smile):
        smile = smile.replace('Cl', 'X')
        smile = smile.replace('Br', 'Y')
        smile = smile.replace('Si', 'G')
        smile = smile.replace('Se', 'Z')
        smile = smile.replace('10', '0')
        smile = smile.replace('Na', 'M')
        smile = smile.replace('Fe','T')
        smile = smile.replace('K','T')
        smile = smile.replace('Au','T')
        smile = smile.replace('As','T')
        smile = smile.replace('Cr','T')
        smile = smile.replace('Al','T')
        smile = smile.replace('[C@@H]','C')
        smile = smile.replace('[C@H]','C')
        smile = smile.replace('[C@@]','C')
        smile = smile.replace('[C@]','C')
        return smile

    def search_smile(self):
        total_search = []
        dict_t = {}
        with open(self.path,'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                if 'source' in line:
                    continue
                else:
                    no = line[0]
                    isomericSMILES = line[2]
                    searchSMILES = line[3]
                    dict_t[no]=isomericSMILES
                    total_search.append(searchSMILES)
            dict_t['1'] = 'ring number 1'
            dict_t['2'] = 'ring number 2'
            dict_t['3'] = 'ring number 3'
            dict_t['4'] = 'ring number 4'
            dict_t['5'] = 'ring number 5'
            dict_t['6'] = 'ring number 6'
            dict_t['7'] = 'ring number 7'
            dict_t['8'] = 'ring number 8'
            dict_t['9'] = 'ring number 9'
            dict_t['10'] = 'C@H'
            dict_t['11'] = 'C@@H'
        return dict_t,total_search

    def get_score(self):
        score = torch.zeros((2190))
        score_temp = []
        with open(self.path,'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                if 'source' in line:
                    continue
                else:
                    score_temp.append(float(line[5]))
        for i in range(11):
            score[i] = 1
        for i,element in enumerate(score_temp):
            score[i+11] = element
            
        return score
        

    def number(self, smile):
        fingerprint = torch.zeros((2190))
        number_temp = []
        for i in smile:
            try:
                int(i)
                number_temp.append(int(i))
            except:
                continue
        if len(number_temp) > 0:
            number_max = max(number_temp)
            assert number_max <= 9, \
                'the maximum of number_max is 9'
            fingerprint[number_max-1] = 1
        if 'C@H' in smile:
            fingerprint[9] = 1
        if 'C@@H' in smile:
            fingerprint[10] = 1
        return fingerprint

    def fun_smile(self,smile,search):
        ele = ['C','H','N','O','P','S','X','Y','I','G','F']
        key = ['-','=','#','\\','/',':','~']
        ele1 = ['N','O','P','S','X','Y','I','G','F']
        sym = ['\d?\(?\)?']
        fingerprint = self.number(smile)
        smile = self.replacex(smile)
        for i,element in enumerate(search):
            if i < 2074:
                substructure = re.findall(r'{0}'.format(element),smile)
                if len(substructure) == 0:
                    continue
                elif len(substructure) >= 1:
                    fingerprint[i+11] = 1
            if i >= 2074:
                substructure = re.findall(r'{0}'.format(element).format(ele,key,ele1,sym),smile)
                if len(substructure) == 0:
                    continue
                elif len(substructure) >= 1:
                    fingerprint[i+11] = 1
        return fingerprint

    def restore(self, smile):
        restore_temp = []
        restore_result = []
        dict_t, total_search = self.search_smile()
        fingerprint = self.fun_smile(smile,total_search)
        for i,element in enumerate(fingerprint):
            if element == 0:
                continue
            elif element == 1:
                restore_temp.append(str(i+1))
        for i in restore_temp:  
            restore_result.append(dict_t[i])
        if 'ring number' not in restore_result[0]:
            restore_temp1 = []
            for i in restore_result:
                restore_search1 = re.findall(r'\d.*\d',i)
                if len(restore_search1) >= 1:
                    continue
                elif len(restore_search1) == 0:
                    restore_temp1.append(i)
            restore_result = restore_temp1
        return restore_result
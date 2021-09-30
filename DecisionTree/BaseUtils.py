class BaseUtils:
    @staticmethod
    def read_dataset(filename):
        """
        年龄段：0代表青年，1代表中年，2代表老年；
        有工作：0代表否，1代表是；
        有自己的房子：0代表否，1代表是；
        信贷情况：0代表一般，1代表好，2代表非常好；
        类别(是否给贷款)：0代表否，1代表是
        """
        fr = open(filename, 'r')
        all_lines = fr.readlines()  # list形式,每行为1个str
        # print(all_lines)
        # print all_lines
        labels = ['年龄段', '有工作', '有自己的房子', '信贷情况']
        dataset = []
        for line in all_lines[0:]:
            line = line.strip().split(',')  # 以逗号为分割符拆分列表
            dataset.append(line)
        return dataset, labels
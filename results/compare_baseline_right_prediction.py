"""
Compare experiment result based on baseline right prediction
"""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare_file', required=True, help='input detail file path')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    baseline_file = "./details_test_1537403767.txt"
    right_qa_id_set = set()
    right_qa_id_all = []
    with open(baseline_file) as fopen:
        for line in fopen:
            line_list = line.strip().split('\001')
            qa_id = int(line_list[0])
            selected = int(line_list[2])
            if selected == 1:
                if qa_id in right_qa_id_set:
                    print(qa_id)
                right_qa_id_set.add(qa_id)
                right_qa_id_all.append(qa_id)

    compare_file = params['compare_file']
    correct_cnt = 0.0
    wrong_cnt = 0.0
    with open(compare_file) as fopen:
        for line in fopen:
            line_list = line.strip().split('\001')
            qa_id = int(line_list[0])
            selected = int(line_list[2])
            if qa_id not in right_qa_id_set:
                continue
            if selected == 1:
                correct_cnt += 1
            else:
                wrong_cnt += 1

print(len(right_qa_id_all))
print('Compare %s to baseline, total:%s, right: %s-%.4f, wrong: %s-%.4f' % (
       compare_file, len(right_qa_id_set), 
       correct_cnt, correct_cnt/len(right_qa_id_set), 
       wrong_cnt, wrong_cnt/len(right_qa_id_set) ))

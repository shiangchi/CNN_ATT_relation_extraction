import codecs
import re
import sys
sys.path.insert(0,'../corpus')
from filter_hypernym import all_file, file_content



def get_pred_label(label_file) :

	pred_label_idx = []
	with open(label_file,'r') as f :
		label = f.readlines()
		pred_label_list = label
		for i in range(len(pred_label_list)):
			if int(pred_label_list[i]) == 1 :
				pred_label_idx.append(i)
				

	print(len(pred_label_idx))
	# print 
	return pred_label_idx

# print(len(pred_label_idx))
def get_both_label(label_file, org_id_file) :
	cid_pred_ans = list()
	with open(label_file,'r') as f :
		pred_label_list = f.readlines()
		# print(pred_label_list)
		for i in range(len(pred_label_list)):
			label = pred_label_list[i].split('\t')[0] 
			pmid = pred_label_list[i].split('\t')[1]
			chem = pred_label_list[i].split('\t')[2]
			dis = pred_label_list[i].split('\t')[3]
			if int(label) == 0 :
				cid_pred_ans.append((pmid, chem, dis))


	test_info = []
	with codecs.open(org_id_file, encoding="utf-8") as f:
		inst_lines = f.readlines()
		for i in range(len(inst_lines)):
			
			parts = inst_lines[i].strip('\n').split("\t")
			if len(parts) == 11 :

				pmid = parts[1]
				chem = parts[2]
				pos1_chem = parts[3]
				dis = parts[4]
				pos2_dis = parts[5]

				test_info.append((pmid, chem, pos1_chem, dis, pos2_dis))

			else :

				chem_dis_men_infos = parts[1].split("_")
				pmid = chem_dis_men_infos[0]
				
				pos1_chem = chem_dis_men_infos[2]
				chem = chem_dis_men_infos[8]
				pos2_dis = chem_dis_men_infos[9]
				dis = chem_dis_men_infos[15]
				test_info.append((pmid, chem, pos1_chem, dis, pos2_dis))


	# print(len(test_info))
	# print(len(cid_pred_ans))
	# print(len(test_info))
	
	pre_pmid_men = {}
	for i in range(len(test_info)) :
		for j in range(len(cid_pred_ans)) :
			test_pmid = test_info[i][0].strip()
			test_chem = test_info[i][1].strip()
			test_chem_id = test_info[i][2].strip()
			test_dis = test_info[i][3].strip()
			test_dis_id = test_info[i][4].strip()

			cid_pred_pmid = cid_pred_ans[j][0].strip()
			cid_pred_chem = cid_pred_ans[j][1].strip()
			cid_pred_dis = cid_pred_ans[j][2].strip()

			if test_pmid == cid_pred_pmid :
				# print('1')
				# print("test_chem", test_chem)
				# print("cid_pred_chem", cid_pred_chem)
				# print("test_dis", test_dis)
				# print("cid_pred_dis", cid_pred_dis)
				if test_chem == cid_pred_chem and test_dis == cid_pred_dis :
					# print("2")
					
					if cid_pred_pmid not in pre_pmid_men.keys() :
						pre_pmid_men[cid_pred_pmid] = [(test_chem_id, test_dis_id)]
					else :
						pre_pmid_men[cid_pred_pmid].append((test_chem_id, test_dis_id))
	
	return pre_pmid_men

def get_ans_label(org_id_file) :

	test_info = list()
	test_comp_info = dict()
	with codecs.open(org_id_file, encoding="utf-8") as f:
		inst_lines = f.readlines()
		for i in range(len(inst_lines)):
			# version for one sentence
			# parts = inst_lines[i].strip('\n').split("\t")
			# # label = parts[0]

			# chem_dis_men_infos = parts[1].split("_")

			# pmid = chem_dis_men_infos[0]
			# pos1_chem = chem_dis_men_infos[2]
			# pos2_dis = chem_dis_men_infos[9]
			# test_info.append((pmid, pos1_chem, pos2_dis))

			# version for two sentence
			# parts = inst_lines[i].strip('\n').split("\t")
			# label = parts[0]
			# pmid = parts[1]
			# pos1_chem = parts[3]
			# pos2_dis = parts[5]
   #          # chem = parts[2]
   #          # dis = parts[4]
			# test_info.append((pmid, pos1_chem, pos2_dis))
			parts = inst_lines[i].strip('\n').split("\t")
			if len(parts) >= 13 :
			# if len(parts) == 11 :

				pmid = parts[1]
				chem_mesh = parts[3]
				dis_mesh = parts[5]
				chem_name = parts[2]
				dis_name = parts[4]
				chem_pos = parts[6]
				dis_pos = parts[7]
				sentence = parts[9]

				test_info.append((pmid, chem_mesh, dis_mesh))
				if pmid not in test_comp_info.keys() :
					test_comp_info[pmid] = list()
					test_comp_info[pmid].append((chem_mesh, dis_mesh, chem_pos, dis_pos, sentence)) # complete information in testing data
				else :
					test_comp_info[pmid].append((chem_mesh, dis_mesh, chem_pos, dis_pos, sentence)) # complete information in testing data


			else :

				chem_dis_men_infos = parts[1].split("_")
				pmid = chem_dis_men_infos[0]
				pos1_chem = chem_dis_men_infos[2]
				pos2_dis = chem_dis_men_infos[9]
				test_info.append((pmid, pos1_chem, pos2_dis))


	print(len(test_info))
	return test_info , test_comp_info

def get_pred_re_ans(pred_label_idx, test_info) :
	
	pre_pmid_men = {}
	for i in pred_label_idx :
		
		pmid = test_info[i][0]
		chem = test_info[i][1]
		dis = test_info[i][2]

		if pmid not in pre_pmid_men.keys() :
			pre_pmid_men[pmid] = [(chem, dis)]

		else :
			pre_pmid_men[pmid].append((chem, dis))
	# print "pre_pmid_men", pre_pmid_men
	return pre_pmid_men


def get_org_abs(org_abs_file) :

	abs_lines = []
	with codecs.open(org_abs_file, encoding="utf-8") as f:
		abs_lines = f.readlines()
		f.close()
	return abs_lines

def get_art_info(abs_lines) :
	art_info = {} 
	for i in range(0,len(abs_lines)) :
		if "|t|" in abs_lines[i]:
			pmid = abs_lines[i].split('|t|')[0]
			title_all = abs_lines[i]
			# print(pmid)
			if "|a|" in abs_lines[i+1]:
				abs_all = abs_lines[i+1]

		art_info[pmid] = (title_all,abs_all)
	return art_info
# print(len(art_info))
def del_same(dict_clean) :
	values = set()
	for pmid in dict_clean.keys():
		val = dict_clean[pmid]
		val_set = set(val)
		dict_clean[pmid] = list(val_set)
	return dict_clean


def remove_hypernym_step1(pre_pmid_men) :
	# print(pre_pmid_men)
	# path = "corpus/MeSH_Disease_no" # in this line it can know if calculate the hypernym term
	path = "../corpus/MeSH_Disease"
	all_disease_tree_num = list()
	filename_list = all_file(path)
	all_disease_tree_num = file_content(path, filename_list, all_disease_tree_num)
	
	tmp_add_mesh2tree = dict()
	clean_add_m2t_old = dict()
	for pmid in pre_pmid_men.keys() :
		# print(pmid)
		# if pmid == '11366874' :
		for chem,dis in pre_pmid_men[pmid] :

			for i in range(len(all_disease_tree_num)) :
				meshid = all_disease_tree_num[i][0]

				if dis in meshid :
					# print("dis", dis)	
					tree_num = all_disease_tree_num[i][1]
					
					if pmid not in tmp_add_mesh2tree.keys() :
						tmp_add_mesh2tree[pmid] = [(chem, dis, tree_num)]
					else :
						tmp_add_mesh2tree[pmid].append((chem, dis, tree_num))

	clean_add_m2t_old = del_same(tmp_add_mesh2tree)
	return clean_add_m2t_old

def	remove_hypernym_step2(pre_pmid_men, clean_add_m2t_old) :
	
	# print(clean_add_m2t_old)
	for pmid in pre_pmid_men.keys() :
		if pmid not in clean_add_m2t_old.keys() :

			no_tree_num_list = pre_pmid_men[pmid]
			clean_add_m2t_old[pmid] = no_tree_num_list

		else:
			for pre_chem, pre_dis in pre_pmid_men[pmid] :
				for i in range(len(clean_add_m2t_old[pmid])) :

					if pre_chem not in clean_add_m2t_old[pmid][i][0] :
						if pre_dis not in clean_add_m2t_old[pmid][i][1] :

							clean_add_m2t_old[pmid].append((pre_chem, pre_dis))
							del_same(clean_add_m2t_old)


	clean_add_m2t = del_same(clean_add_m2t_old)
	return clean_add_m2t

def remove_hypernym_step3(clean_add_m2t) :

	need_to_del = dict()

	for pmid in clean_add_m2t.keys() :
		for i in range(len(clean_add_m2t[pmid])) :

			if len(clean_add_m2t[pmid][i]) == 3 :
				chem = clean_add_m2t[pmid][i][0]
				dis = clean_add_m2t[pmid][i][1]
				tree = clean_add_m2t[pmid][i][2]

				
				for j in range(i+1,len(clean_add_m2t[pmid])) :
					if len(clean_add_m2t[pmid][j]) == 3 :					
						next_chem = clean_add_m2t[pmid][j][0]
						next_dis = clean_add_m2t[pmid][j][1]
						next_tree = clean_add_m2t[pmid][j][2]

						if tree.find(next_tree) != -1 :
							if chem == next_chem :
								if pmid not in need_to_del.keys() :
									need_to_del[pmid] = [(next_chem, next_dis, next_tree)]
								else :
									need_to_del[pmid].append((next_chem, next_dis, next_tree))

						if next_tree.find(tree) != -1 :
							if chem == next_chem :
								if pmid not in need_to_del.keys() :
									need_to_del[pmid] = [(chem, dis, tree)]
								else :
									need_to_del[pmid].append((chem, dis, tree))
								
	clean_need_to_del = del_same(need_to_del)
	return clean_need_to_del
	
def remove_hypernym_step4(clean_need_to_del, clean_add_m2t) :
	no_hypernym_pre_pmid_men = dict()

	for pmid in clean_need_to_del.keys() :
		del_list = clean_need_to_del[pmid]
		org_list = clean_add_m2t[pmid]
		# print(pmid)
		# print("org_list", org_list)
		res_list = list(set(org_list).difference(set(del_list)))
		# print("res_list", res_list)
		
		if pmid not in no_hypernym_pre_pmid_men.keys() :
			no_hypernym_pre_pmid_men[pmid] = list()
			for i in range(len(res_list)) :
				chem = res_list[i][0]
				dis = res_list[i][1]
				no_hypernym_pre_pmid_men[pmid].append((chem, dis))
	# print("1", no_hypernym_pre_pmid_men)

	# return no_hypernym_pre_pmid_men
# def remove_hypernym_step5(no_hypernym_pre_pmid_men, clean_add_m2t) :
	for pmid in clean_add_m2t.keys() :

		add_list = clean_add_m2t[pmid]
		if pmid not in no_hypernym_pre_pmid_men.keys() :
			no_hypernym_pre_pmid_men[pmid] = list()
			for i in range(len(add_list)) :
				chem = add_list[i][0]
				dis = add_list[i][1]
				no_hypernym_pre_pmid_men[pmid].append((chem, dis))
					
	# print("2",no_hypernym_pre_pmid_men)
	return no_hypernym_pre_pmid_men

# print(all_disease_tree_num)
def split_abs(abstract) :
	abstract = abstract.strip()
	abstract = abstract.replace("(","( ")
	abstract = abstract.replace(")"," )")
	abstract = abstract.replace(","," ,")
	abstract = abstract.replace(";"," .") # add for _new version
	abstract = abstract.replace("!"," .") # add for _new version

	# abstract_dash = re.findall(r'(\w\-indcued)|(\s\-\w)|(\w\-\s)',abstract)
	abstract_UP = re.findall(r'([A-Z]+[:]+[ ])',abstract)
	for i in range(len(abstract_UP)) :
		abstract = abstract.replace(abstract_UP[i],"")

	abstract_dash = re.findall(r'(\w\-induced)',abstract)
	

	if len(abstract_dash) > 0 :
		for i in range(len(abstract_dash)):
			for idx, ele in enumerate(abstract_dash[i]) :
				if ele !=  '' :
					ans = abstract_dash[i].replace('-',' -')
					abstract = abstract.replace(abstract_dash[i],ans)
					

	abstract = abstract.replace(".( ABSTRACT TRUNCATED AT 250 WORDS )\n"," ")
	# abstract_split = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z][0-9]\.)(?<=\.|\?|\!)\s*(?=[A-Z])',abstract)
	# abstract_split = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z][0-9]\.)(?<=\.|\?|\!)\s*(?=[A-Z]|[(])',abstract)
	abstract_split = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z][0-9]\.)(?<=\.|\?)\s*(?=[A-Z]|[(])',abstract)

	return abstract_split

def sentence_node(sentence):
	# for i in range(len(abstract_split)) :
	# if ' ;' not in sentence :
	s = re.findall(r'[\D|\d][\.|\;|?]',sentence) # did not consider number
	# print s
	# if ';' in sentence :
	# 	print sentence 
	if s[-1][0] != ' ' :
		sentence = sentence.replace(s[-1], s[-1][0] +' '+ s[-1][1])
	
	else :
		sentence = sentence

	
	return sentence.lower()
# def rule_based(art_info, no_hypernym_pre_pmid_men, test_comp_info) : # this method is not good, too.
# 	negatives = ["no",'not','without',"neither", 'any', 'nor', 'avoid']
# 	for pmid in test_comp_info.keys() :
# 		for chem_mesh, dis_mesh, chem_pos, dis_pos, sentence in test_comp_info[pmid] :
# 			if "side effect" in sentence or "adverse event" in sentence :
# 				if int(chem_pos) < int(dis_pos) and dis_mesh != "D006402" and dis_mesh != "D014657" \
# 				and dis_mesh != "D001943" and dis_mesh != "D010146" :
# 					abst = art_info[pmid][1].split('|a|')[1]

# 					abst_sp = split_abs(abst)
# 					# print abst_sp
# 					# sentence_sp = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z][0-9]\.)(?<=\.|\?)\s*(?=[A-Z]|[(])',sentence)
# 					sentence_sp = sentence.split(' . ')
					
# 					# print len(sentence_sp)
# 					for i in range(len(abst_sp)) :
# 						if i+1 < len(abst_sp) :
# 							sent_1 = sentence_node(abst_sp[i])
# 							sent_2 = sentence_node(abst_sp[i+1])
							
# 							# if sent_1 == sentence_sp[0] and sent_2 == sentence_sp[1] :
# 							if sentence_sp[0] in sent_1 and sent_2 == sentence_sp[1] :


# 								if pmid not in no_hypernym_pre_pmid_men.keys() :
# 									no_hypernym_pre_pmid_men[pmid] = list()
# 									no_hypernym_pre_pmid_men[pmid].append((chem_mesh, dis_mesh))
# 								else :
# 									no_hypernym_pre_pmid_men[pmid].append((chem_mesh, dis_mesh))



# 	return no_hypernym_pre_pmid_men
# if tagger_c_index >= 4 :

# 				re_sentence_sp = sentence_sp[tagger_c_index-4 : tagger_c_index+5]
				
# 				if "side effect" in sentence or "adverse event" in sentence :
					
# 					for negative in negatives :
# 						if negative in re_sentence_sp :
# 							tagger_word_y_n = 0

# 							break
# 						else :
# 							tagger_word_y_n = 1
# def rule_based(no_hypernym_pre_pmid_men, test_comp_info) : this version for add negative word around entity, but not useful
# 	no_hypernym_pre_pmid_men = dict()
# 	negatives = ["no",'not','without',"neither", 'any', 'nor', 'avoid']
# 	for pmid in test_comp_info.keys() :
# 		for chem_mesh, dis_mesh, chem_pos, dis_pos, sentence in test_comp_info[pmid] :
# 			if "side effect" in sentence or "adverse event" in sentence :
# 				if int(chem_pos) < int(dis_pos) and dis_mesh != "D006402" and dis_mesh != "D014657" \
# 				and dis_mesh != "D001943" and dis_mesh != "D010146" :
# 					sentence_sp = sentence.split(' ')
# 					if int(chem_pos) >= 4 :
# 						re_s_sp = sentence_sp[int(chem_pos)-4 : int(chem_pos)+5]
# 						for negative in negatives :
# 							if negative in re_s_sp :
# 								pass
# 							else :

# 								if pmid not in no_hypernym_pre_pmid_men.keys() :
# 									no_hypernym_pre_pmid_men[pmid] = list()
# 									no_hypernym_pre_pmid_men[pmid].append((chem_mesh, dis_mesh))
# 								else :
# 									no_hypernym_pre_pmid_men[pmid].append((chem_mesh, dis_mesh))



# 	return no_hypernym_pre_pmid_men

def rule_based(no_hypernym_pre_pmid_men, test_comp_info) :
	# no_hypernym_pre_pmid_men = dict()
	
	for pmid in test_comp_info.keys() :
		for chem_mesh, dis_mesh, chem_pos, dis_pos, sentence in test_comp_info[pmid] :
			# if "side effect" in sentence or "adverse event" in sentence : 
			if "side effect" in sentence :
				if int(chem_pos) < int(dis_pos) and dis_mesh != "D006402" and dis_mesh != "D014657" \
				and dis_mesh != "D001943" and dis_mesh != "D010146" :

					if pmid not in no_hypernym_pre_pmid_men.keys() :
						no_hypernym_pre_pmid_men[pmid] = list()
						no_hypernym_pre_pmid_men[pmid].append((chem_mesh, dis_mesh))
					else :
						no_hypernym_pre_pmid_men[pmid].append((chem_mesh, dis_mesh))

	return no_hypernym_pre_pmid_men

def final_trans(art_info, no_hypernym_pre_pmid_men, write_file) :
	
	# write_file = 'test.txt'
	with open(write_file,'w') as out :		

		for pmid in art_info.keys() :
			title = art_info[pmid][0]
			abst = art_info[pmid][1]
			out.write(title)
			out.write(abst)
			
			if pmid not in no_hypernym_pre_pmid_men.keys() :
				out.write('\n')

			else :
				pre_pmid_men_set = set(no_hypernym_pre_pmid_men[pmid])
				pre_pmid_men_list = list(pre_pmid_men_set)
				# print("pre_pmid_men_list",pre_pmid_men_list)

				for i in range(len(pre_pmid_men_list)) :
					out.write(pmid)
					out.write('\t')
					out.write('CID')
					chem = pre_pmid_men_list[i][0]
					dis = pre_pmid_men_list[i][1]

					out.write('\t')
					out.write(chem)
					out.write('\t')
					out.write(dis)
					out.write('\n')
					
				out.write('\n')
		out.close()

if __name__ == '__main__':
	# label_file = 'pred_result/cross_v3_1:7_best_record.txt'
	# label_file ='pred_result/best_record.txt'
	# label_file = "pred_result/predict_label_s"
	# label_file = "pred_result/lstm_across_withclose_verb_v2"
	# label_file = "pred_result/lstm_across_add_coref"
	# label_file = "pred_result/cnn_across_close_verb_del_same"
	label_file = "../pred_result/cnn_across_only_across"
	# label_file = "pred_result/lstm_cross_newpos"
	# label_file = "/home/hsu/Downloads/nary-grn-master/gs_lstm/LSTM_CNN_CDR_both/result_LCboth"
	# label_file = "/home/hsu/Downloads/nary-grn-master/gs_lstm/LSTM_CDR/result_both"

	# org_id_file = 'files/testSet.instances'
	# org_id_file = 'corpus/CDR_TestSet.PubTatoraddshortest_to2_v3.txt'
	# org_id_file = 'corpus/data_prepro/CDR_TestSet_change_abb2full_addcoref_2.txt'
	# org_id_file = 'corpus/mult_feat/CDR_TestSet.PubTator_similar_v2_del_same'
	# org_id_file = 'corpus/mult_feat/test_only_across_relation_v2'
	# org_id_file = '../corpus/mult_feat/CDR_TestSet.PubTator_new_2_del_same_3'
	# org_id_file = '../corpus/mult_feat/CDR_TestSet.PubTator_new_4_v6_del_same_7'
	org_id_file = '../corpus/mult_feat/new_single_test'
	# org_id_file = 'corpus/data_prepro/close_verb/del_same/CDR_TestSet.PubTator_close_verb_list_del_same.txt'
	# org_id_file = "corpus/All_TE"

	org_abs_file = '../corpus/gold_data/CDR_TestSet.PubTator.txt'

	
	# # LSTM WAY
	# pre_pmid_men = get_both_label(label_file, org_id_file)

	# CNN WAY
	pred_label_idx = get_pred_label(label_file)
	
	test_info , test_comp_info  = get_ans_label(org_id_file)
	
	pre_pmid_men = get_pred_re_ans(pred_label_idx, test_info)


	abs_lines = get_org_abs(org_abs_file)
	art_info = get_art_info(abs_lines) # get the title and abstract 
	# print(pre_pmid_men)
	# print(len(pre_pmid_men))
	clean_add_m2t_old = remove_hypernym_step1(pre_pmid_men)
	clean_add_m2t = remove_hypernym_step2(pre_pmid_men, clean_add_m2t_old)
	clean_need_to_del = remove_hypernym_step3(clean_add_m2t)
	no_hypernym_pre_pmid_men = remove_hypernym_step4(clean_need_to_del, clean_add_m2t)

	pred_re_pair = rule_based(no_hypernym_pre_pmid_men, test_comp_info) # pred_re_pair = predict_relation_pair



	# print(no_hypernym_pre_pmid_men)
	# write_file = 'add_pos_result_remove_hypernym.Pubtator_to2_v2_1:13.txt'
	# write_file = 'new_model_ans_s'
	# write_file = 'LSTM_'+label_file.split('/')[-1].split('_')[1]
	write_file = '../convert2pub/' + label_file.split('/')[-1]
	# final_trans(art_info, no_hypernym_pre_pmid_men, write_file)
	final_trans(art_info, pred_re_pair, write_file)



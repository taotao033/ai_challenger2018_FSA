import re

# str = '"劳斯莱斯 寿司 卷'
# str = re.sub("\"", " ", str)
#
# print(str.strip())

#if __name__ == "__main__":

    # train_file = './dataset/data_reform/train_reform_content_after_cut_mini.csv'
	# x, y, vocab, vocab_inv, df, labels = load_data(train_file, max_length=1000)
	# df = pd.read_csv("./dataset/valid.csv", encoding="utf-8")
	# content = df["content"]
	# content_list = seg_word(content)
	# df["content"] = content_list
	# df.to_csv("./dataset/valid_content_after_cut.csv", index=False, sep=",", encoding="utf-8")
	#
	# balance_data_dict = get_balance_train_data(path="./dataset/data_reform/train_reform_content_after_cut.csv")
	# data_frame = pd.DataFrame({"content": balance_data_dict["location_traffic_convenience"][0],
	# 						"location_traffic_convenience": balance_data_dict["location_traffic_convenience"][1]})
	#
	# data_frame.to_csv("./dataset/data_reform/balance_location_traffic_convenience.csv", index=False, sep=",", encoding="utf-8")
	#
	# df = pd.read_csv("./dataset/data_reform/balance_location_traffic_convenience.csv", encoding="utf-8", header=None)
	# ds = data_frame.sample(frac=1)
	# ds.to_csv("new_files_balance.csv", index=False, sep=",", encoding="utf-8")
	#
	# print("the number of -2: " + str(len(ds[ds["location_traffic_convenience"] == -2]["location_traffic_convenience"])))
	# print("the number of -1: " + str(len(ds[ds["location_traffic_convenience"] == -1]["location_traffic_convenience"])))
	# print("the number of 0: " + str(len(ds[ds["location_traffic_convenience"] == 0]["location_traffic_convenience"])))
	# print("the number of 1: " + str(len(ds[ds["location_traffic_convenience"] == 1]["location_traffic_convenience"])))
	# print(str(len(ds[ds["location_traffic_convenience"] == "location_traffic_convenience"])))
	# df = pd.read_csv("./new_files_update.csv", encoding="utf-8")
	# #print(len(df[df["location_traffic_convenience"] == -2]))
	# print(len(df[df["location_traffic_convenience"] == 0]))
	# print(len(df[df["location_traffic_convenience"] == -1]))
	#df.to_csv("./new_files_update.csv", index=False, sep=",", encoding="utf-8")
import json

with open("vocab.json", encoding="utf-8") as f:
    vocabulary = json.load(f)
    print(vocabulary)
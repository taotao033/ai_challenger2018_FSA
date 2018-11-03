import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


key_list1 = [
    "location_traffic_convenience",
    "location_distance_from_business_district",
    "location_easy_to_find",
    "service_wait_time",
    "service_waiters_attitude",
    "service_parking_convenience",
    "service_serving_speed",
    "price_level",
    "price_cost_effective",
    "price_discount",
    "environment_decoration",
    "environment_noise",
    "environment_space",
    "environment_cleaness",
    "dish_portion",
    "dish_taste",
    "dish_look",
    "dish_recommendation",
    "others_overall_experience",
    "others_willing_to_consume_again"
]


def f1_macro():
    df = pd.read_csv("./output/val_predicted.csv", encoding='utf-8')  # predicted
    df_val = pd.read_csv("./dataset/valid.csv", encoding='utf-8')  # really

    f1_per_column = open("./output/f1_score_cnn_rnn_val.txt", 'a+')
    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for column in key_list1:
        accuracy = accuracy_score(df_val[column], df[column], normalize=True)
        precision = precision_score(df_val[column], df[column], average='macro')
        recall = recall_score(df_val[column], df[column], average='macro')
        f1 = f1_score(df_val[column], df[column], average='macro')
        f1_per_column.write("f1_score_" + column + ": " + str(f1) + "\n")

        accuracy_sum += accuracy
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    average_accuracy = accuracy_sum / len(key_list1)
    average_precision = precision_sum / len(key_list1)
    average_recall = recall_sum / len(key_list1)
    average_f1 = f1_sum / len(key_list1)
    f1_per_column.write("\n" + 'average_accuracy: ' + str(average_accuracy) + '\n')
    f1_per_column.write('average_precision: ' + str(average_precision) + '\n')
    f1_per_column.write('average_recall: ' + str(average_recall) + '\n')
    f1_per_column.write('average_f1: ' + str(average_f1))
    f1_per_column.close()
    if average_f1 >= 0.750:
        print("Great! Nice! you win!")
    else:
        print("Unfortunately,come on!")
    print('f1_score: ' + str(average_f1))


f1_macro()

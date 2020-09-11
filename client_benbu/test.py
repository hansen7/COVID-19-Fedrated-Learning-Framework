
import os
import sys
import csv
import pdb
import json
import torch
import numpy as np
import pandas as pd
sys.path.append('common')
from logger import create_log
import torch.nn.functional as F
from data_raw import TestDataset
from scipy.special import softmax
from torch.nn import DataParallel
from model.model import densenet3d
from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, \
    confusion_matrix, precision_score, recall_score, f1_score

'''Uncomment this if use specified GPUs'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size


def results(labels, preds, avg_type, class_type="other"):
    precision = precision_score(labels, preds, average=avg_type)
    recall = recall_score(labels, preds, average=avg_type)
    f1score = f1_score(labels, preds, average=avg_type)
    log.info("class {} precision:{:.4f} recall:{:.4f} f1 score:{:.4f} ".format(
        avg_type, precision, recall, f1score))
    report = classification_report(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    log.info(f'\n{report}\n{conf_matrix}\n')
    if class_type == "two":
        return recall


class Prediction:
    def __init__(self, outputs, labels, path_name, patient_id):
        self.outputs = outputs
        self.labels = labels
        self.path_name = path_name
        self.patient_id = patient_id

    def __eq__(self, other):
        if self.patient_id == other.pateint_id:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.patient_id != other.patient_id:
            return self.patient_id > other.patient_id
        else:
            return self.path_name > other.path_name


def gen_dict(pred):
    """"""
    pred_sorted = sorted(pred)
    pred_lists = [[pred_sorted[0]]]
    for i in range(1, len(pred_sorted)):
        cur_info, pre_info = pred_sorted[i], pred_sorted[i - 1]
        if cur_info.patient_id != pre_info.patient_id:
            pred_lists.append([cur_info])
        else:
            pred_lists[-1].append(cur_info)
    return pred_lists


def gen_two_class(preds, labels):
    label_two, pred_two = [], []

    for label in labels:
        if label == 1 or label == 4 or label == 5:
            label_two.append(1)
        else:
            label_two.append(0)

    for pred in preds:
        if pred == 1 or pred == 4 or pred == 5:
            pred_two.append(1)
        else:
            pred_two.append(0)
    return pred_two, label_two


def gen_four_class(preds, labels):
    """range from 0, 1, 2, 3, 4, 5"""
    label_four, pred_four = [], []

    for label in labels:
        if label == 1 or label == 4 or label == 5:
            label_four.append(1)
        else:
            label_four.append(label)

    for pred in preds:
        if pred == 1 or pred == 4 or pred == 5:
            pred_four.append(1)
        else:
            pred_four.append(pred)
    return pred_four, label_four


def test(test_data_loader, model, patient_id_list):
    predictions_allmodels, all_output_value = [], []

    for _ in range(5):
        predictions, epoch_output_value = [], []

        for index, (inputs, labels, patient_name, patient_ids) in enumerate(test_data_loader):
            model.eval()
            with torch.no_grad():
                inputs = inputs.cuda().unsqueeze(dim=1).float()
                inputs = F.interpolate(inputs, size=[16, 128, 128],
                                       mode="trilinear", align_corners=False)
                outputs = model(inputs)  # (batch size, num class)
                outputs_array = softmax(outputs.detach().cpu().numpy(), axis=1)
                for oo in outputs_array:
                    epoch_output_value.append(oo)
                acc = calculate_accuracy(outputs, labels.cuda())
                log.info('batch: {}, acc: {}'.format(index + 1, acc))

            for index, patient_id in enumerate(patient_ids):
                if patient_id in patient_id_list:
                    prediction = Prediction(outputs_array[index], labels[index],
                                            patient_name[index], patient_id)
                    predictions.append(prediction)
        predictions_allmodels.append(predictions)
        all_output_value.append(epoch_output_value)

    all_output_value = np.asarray(all_output_value).mean(0)
    print('all output value shape', all_output_value.shape)
    npa_predictions = np.asarray(predictions_allmodels)
    print('npa predictions shape', npa_predictions.shape)  # (num class, num test cases)

    predictions_final, all_output = [], []
    for index in range(npa_predictions.shape[1]):
        npa_predict = npa_predictions[:, index]
        outputs_value = []
        for predict_value in npa_predict:
            outputs_value.append(predict_value.outputs)
        outputs_value = np.asarray(outputs_value)
        outputs_value = outputs_value.mean(0)
        all_output.append(outputs_value)

        prediction = Prediction(outputs_value, npa_predict[0].labels,
                                npa_predict[0].path_name, npa_predict[0].patient_id)
        predictions_final.append(prediction)

    return predictions_final


if __name__ == "__main__":
    with open('./config/train_config_client.json') as j:
        train_config = json.load(j)
    data_test = TestDataset(train_config['test_data_dir'],
                            train_config['test_df_csv'],
                            train_config['labels_test_df_csv'])
    test_data_loader = DataLoader(dataset=data_test, batch_size=40,
                                  shuffle=False, num_workers=24)

    os.makedirs('test', mode=0o777, exist_ok=True)
    logfile = r"test/test_cambridge.log"
    log = create_log(logfile)
    # sys.stdout = Logger(logfile)
    # patient_ids = gen_ids("./utils/patients_id_test.csv")
    patient_ids = pd.read_csv(train_config['test_df_csv'])['name'].values

    # for epoch in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
    for epoch in range(5, 100, 5):
        # PATH = "./model/Cambridge_current_{}.pth".format(epoch)
        PATH = "./model/{}_local_central.pth".format(epoch)
        checkpoint = torch.load(PATH)
        model = densenet3d().cuda()
        model = DataParallel(model)
        model.load_state_dict(checkpoint)

        preds = test(test_data_loader, model, patient_ids)
        pred_lists = gen_dict(preds)
        case_preds, case_labels, case_ids, case_path = [], [], [], []
        output_sequence, output_mean = [], []

        for case_pred in pred_lists:
            seq_preds = []
            for seq_pred in case_pred:
                seq_preds.append(seq_pred.outputs)
                sequence_output = np.argmax(seq_pred.outputs)
                label = seq_pred.labels
                patient_id = seq_pred.patient_id
                path_name = seq_pred.path_name
                output_sequence.append([seq_pred.outputs, sequence_output, label,
                                        patient_id, path_name])
                # print('output_sequence', output_sequence)

            mean_pred = np.mean(seq_preds, 0)
            type_pred = np.argmax(mean_pred)
            case_preds.append(type_pred)
            case_ids.append(patient_id)
            case_path.append(path_name)
            case_labels.append(label)
            output_mean.append([mean_pred, type_pred, label, patient_id, path_name])
            # print('output mean', output_mean)

        """ === raw softmax output over 5 runs ==="""
        with open('test/output_sequence5_softmax.csv', 'w')as file:
            spamWriter = csv.writer(file)
            spamWriter.writerows(output_sequence)

        """=== mean of softmax output over 5 runs of each model ==="""
        with open(r'test/output_mean4_softmax.csv', 'w') as file:
            spamWriter = csv.writer(file)
            spamWriter.writerows(output_mean)
        pred_two, label_two = gen_two_class(case_preds, case_labels)
        pred_four, label_four = gen_four_class(case_preds, case_labels)
        box_train = zip(case_path, case_ids, label_four, pred_four, label_two, pred_two)

        with open(r'test/case_test_four_norm{}.csv'.format(epoch), 'w') as result_file:
            wr = csv.writer(result_file)
            wr.writerow(['case path', 'case id', 'label four',
                         'pred four', 'label two', 'pred two'])
            for row in box_train:
                wr.writerow(row)
        # results(label_six, pred_six, avg_type="macro")
        # results(label_six, pred_six, avg_type="micro")
        results(label_four, pred_four, avg_type="macro")
        results(label_four, pred_four, avg_type="micro")
        results(label_two, pred_two, avg_type="macro", class_type="two")

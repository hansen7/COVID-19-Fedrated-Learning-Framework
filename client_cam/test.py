
import os
import csv
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.special import softmax
from torch.nn import DataParallel
from model.model import densenet3d
from torch.utils.data import DataLoader
from common import create_log, TestDataset
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
    # about average type, see:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    precision = precision_score(labels, preds, average=avg_type)
    recall = recall_score(labels, preds, average=avg_type)
    f1score = f1_score(labels, preds, average=avg_type)
    log.info("average type {}, precision:{:.4f} recall:{:.4f} f1 score:{:.4f} ".format(
        avg_type, precision, recall, f1score))
    report = classification_report(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    log.info(f'\n{report}\n{conf_matrix}\n')
    if class_type == "two":
        return recall


class Prediction:
    def __init__(self, outputs, labels, path_name, patient_id):
        self.labels = labels
        self.outputs = outputs
        self.path_name = path_name
        self.patient_id = patient_id

    def __eq__(self, other):
        return self.patient_id == other.pateint_id

    def __gt__(self, other):
        if self.patient_id != other.patient_id:
            return self.patient_id > other.patient_id
        else:
            return self.path_name > other.path_name


def gen_dict(pred):
    """aggregate predictions for the same patient"""
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
    converter = lambda x: 1 if x == 1 else 0
    return list(map(converter, preds)), list(map(converter, labels))


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

    npa_predictions = np.asarray(predictions_allmodels)
    # print('npa predictions shape', npa_predictions.shape)  # (num class, num test cases)

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
    
    logdir = 'test_tongji_xiehe_fl'
    patient_ids = pd.read_csv(train_config['test_df_csv'])['name'].values
    logfile = os.path.join(logdir, "test_cambridge.log")
    os.makedirs(logdir, mode=0o777, exist_ok=True)
    log = create_log(logfile)

    #for epoch in range(5, 100, 5):
    for epoch in ['tongji_xiehe_fl']:
        log.info("\n\n")
        log.info("=" * 33)
        log.info("Use Checkpoints from Epoch {}".format(epoch))
        # PATH = "./model/{}_local_central.pth".format(epoch)
        PATH = "./ref_model/{}.pth".format(epoch)
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
        with open(os.path.join(logdir, 'output_sequence5_softmax.csv'), 'w')as file:
            spamWriter = csv.writer(file)
            spamWriter.writerows(['\n'])
            spamWriter.writerows(['Epoch: %s' % str(epoch)])
            spamWriter.writerows(output_sequence)

        """=== mean of softmax output over 5 runs of each model ==="""
        with open(os.path.join(logdir, 'output_mean4_softmax.csv'), 'w') as file:
            spamWriter = csv.writer(file)
            spamWriter.writerows(['\n'])
            spamWriter.writerows(['Epoch: %s' % str(epoch)])
            spamWriter.writerows(output_mean)
        pred_four, label_four = case_preds, case_labels
        pred_two, label_two = gen_two_class(case_preds, case_labels)
        box_train = zip(case_path, case_ids, label_four, pred_four, label_two, pred_two)

        with open(os.path.join(logdir, 'case_test_four_norm{}.csv'.format(epoch)), 'w') as result_file:
            wr = csv.writer(result_file)
            wr.writerow(['case path', 'case id', 'label four',
                         'pred four', 'label two', 'pred two'])
            for row in box_train:
                wr.writerow(row)
        
        results(label_four, pred_four, avg_type="macro")
        results(label_four, pred_four, avg_type="micro")
        results(label_two, pred_two, avg_type="macro", class_type="two")


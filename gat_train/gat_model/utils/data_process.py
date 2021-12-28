

import sys
sys.path.insert(0,'/mnt/westworld/sunzhe/code/rc_reckon_train')
import json
import numpy as np
from gat_online_train.bean.GraphInfo import EcGraphInfo
from gat_online_train.sparse_gat.utils.norm_util import NormUtil


class GraphBuild(object):
    def __init__(self, mode):
        self.edge_dict = self.genConCategory()
        self.loadNormFeat(mode)

    def genConCategory(self):
        ori_edge_set = [
            "Did",
            "ID_IDFA",
            "ID_IDFV",
            "ID_SecDeviceID",
            "address",
            "Net_Bssid",
            "receiver_mobile",
            "Open_id"
        ]
        count = 0
        edge_dict = {}
        for item in ori_edge_set:
            edge_dict[item] = count
            count += 1
        return edge_dict

    def loadNormFeat(self, mode):
        if mode == "node":
            self.mean = NormUtil.gat_node_mean
            self.std = NormUtil.gat_node_std
        elif mode == "node_v1":
            self.mean = NormUtil.gat_node_v1_mean
            self.std = NormUtil.gat_node_v1_std
        elif mode == "graph":
            self.mean = NormUtil.gat_graph_mean
            self.std = NormUtil.gat_graph_std
        elif mode == "graph_v1":
            self.mean = NormUtil.gat_graph_v1_mean
            self.std = NormUtil.gat_graph_v1_std
        elif mode == "graph_aboard":
            self.mean = NormUtil.gat_node_aboard_mean
            self.std = NormUtil.gat_node_aboard_std
        pass

    def fillFeaArr(self, feature, fea_index=18):
        fea_temp = []
        for col in range(fea_index):
            col = str(col)
            if col in feature:
                if int(col) > fea_index:
                    continue
                fea_temp.append(feature[col])
            else:
                fea_temp.append(0)
        fea_temp = (fea_temp - self.mean) / self.std
        fea_temp[np.isnan(fea_temp)] = 0
        return fea_temp

    def train_val_test_split(self, data_path, train_ratio=0.8, batch_size=5):
        samples = []
        total_cnt = 0
        with open(data_path) as origin_data:
            for sample in origin_data.readlines():
                label, center_uid, graph = sample.split('\t')
                samples.append([label, center_uid, graph])
                total_cnt += 1
        np.random.shuffle(samples)
        train_size = int(train_ratio * total_cnt)
        test_size = total_cnt - train_size

        train_data = samples[0: train_size]
        test_data = samples[train_size: total_cnt]

        train_batches = []
        test_batches = []
        for i in range(int(train_size / batch_size) + 1):
            if i * batch_size >= train_size:
                continue
            train_batch = self.genBatchGraph(train_data[i * batch_size: (i + 1) * batch_size], True)
            train_batches.append(train_batch)

        for i in range(int(test_size / batch_size) + 1):
            if i * batch_size >= test_size:
                continue
            test_batch = self.genBatchGraph(test_data[i * batch_size: (i + 1) * batch_size], True)
            test_batches.append(test_batch)
        print('len(train_batches), len(train_batches)', len(train_batches), len(test_batches))
        return [train_batches, test_batches]

    def genBatchGraph(self, graphs, isTrain=True):
        index = 0
        node_index_map = {}  # uid->index
        user_feat_list = []  # user_feature
        node_dict = {}  # center_uid index

        # load batch all nodes and indexed
        cnt = 0
        l = u = h = 0
        if len(graphs) >= 1:
            l,u,h = graphs[0]
        for sample in graphs:
            label, center_uid, graph = sample
            if graph == '' or graph == '{}' or graph == 'NULL\t' or graph == 'relation\t':
                continue
            try:
                x = json.loads(graph)
            except:
                print(graph)
                continue

            ec_graph = EcGraphInfo(**x)
            if ec_graph is None or ec_graph.Nodes is None or ec_graph.Relations is None:
                continue
            # load node feature
            for uid, feature in ec_graph.Nodes.items():
                if feature is None or feature == '':
                    continue

                # if uid in node_index_map:
                #     continue

                item = json.loads(feature)

                if 'feat_norm' not in item:
                    continue

                feat = item['feat_norm']
                if feat == "" or feat == '{}':
                    continue

                feat = json.loads(feat)
                cnt+=feat['13']
                del feat['86']
                adv_feat = self.fillFeaArr(feat)
                node_index_map[uid] = index
                user_feat_list.append(adv_feat)
                index += 1

            if center_uid not in node_index_map:  # 去除无效图，中心结点无特征
                continue
        # load batch node graph relation and indexed
        conn_feat_list = []  # relation
        adj_mat = []
        edge_set = set([])

        for sample in graphs:  # 遍历每张图
            label, center_uid, graph = sample

            if center_uid not in node_index_map:  # 去除无效图
                continue

            x = json.loads(graph)
            ec_graph = EcGraphInfo(**x)
            for r in ec_graph.Relations:  # 遍历每张图中的关系
                edge1 = r.V0
                rel_node = r.V1
                if '#' not in edge1:
                    edge1 = 'Did#' + edge1

                # if rel_node == center_uid:  # 去除与中心节点连接的一跳边，二跳会包含
                #     continue

                if rel_node not in node_index_map:  # del uid which has not feature
                    continue

                node_index = node_index_map[rel_node]  # connected uid index

                if edge1 not in node_index_map:  # if edge not in node_index, add it
                    node_index_map[edge1] = index
                    conn_feat_list.append([self.edge_dict[edge1.split("#")[0]]])

                    index += 1

                # print(node_index_map)

                edge = node_index_map[edge1]  # connected edge
                key_node = node_index_map[center_uid]
                key_edge1 = "%s%s" % (key_node, edge)  # center_uid, edge

                if key_edge1 not in edge_set:
                    key_edge2 = "%s%s" % (edge, key_node)
                    edge_set.add(key_edge1)
                    edge_set.add(key_edge2)
                    adj_mat.append([key_node, edge])
                    adj_mat.append([edge, key_node])

                add_edge1 = "%s%s" % (edge, node_index)  # connected edge, connected uid
                if add_edge1 not in edge_set:
                    add_edge2 = "%s%s" % (node_index, edge)
                    # print('add_edge2:', node_index, edge)
                    edge_set.add(add_edge1)
                    edge_set.add(add_edge2)
                    adj_mat.append([edge, node_index])
                    adj_mat.append([node_index, edge])

                center_uid_index = node_index_map[center_uid]
                node_dict[center_uid_index] = int(label)

        adj_mat.sort(key=lambda x: [x[0], x[1]], reverse=False)

        value = np.ones(len(adj_mat))
        max_index = index


        if isTrain:
            mask = np.zeros(max_index)  # 是为了标记后边，那些结点用于计算loss
            labels = np.zeros(max_index)
            for center_uid_index, label in node_dict.items():
                mask[center_uid_index] = 1  # 每张图的中心节点为1，其他的都为0，只有中心节点计算loss
                labels[center_uid_index] = int(label)  # 每张图的中心节点为1
            label_list = []
            for item in labels:
                if item == 1:
                    label_list.append((0, 1))
                else:
                    label_list.append((1, 0))

            labels = np.array(label_list).reshape((len(labels), 2))

            return np.array(adj_mat), (max_index, max_index), value, max_index, np.array(user_feat_list)[
                np.newaxis], np.array(conn_feat_list)[np.newaxis], np.array(labels)[np.newaxis], \
                   np.array(mask, dtype=np.bool)[np.newaxis], node_dict, cnt,u


if __name__ == '__main__':
    build = GraphBuild("node_v1")
    train, test = build.train_val_test_split('../data/train_gat.data')
    print(train[0])
    pass

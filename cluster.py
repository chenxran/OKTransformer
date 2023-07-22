from collections import defaultdict
from random import uniform
from math import sqrt, floor
import math
import numpy as np
import random
import copy
# from heap import MinHeap
import numpy as np
# from cluster_cxr import kmeans
import torch
import sklearn as scikit
from scipy.sparse import csr_matrix
import time
from tqdm import tqdm

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, dataset):
        new_means[assignment].append(point)

    for points in new_means.itervalues():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point.
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `dataset` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = ()  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments




def generate_k(dataset, k):
    """
    Given `dataset`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(dataset[0])
    min_max = defaultdict(int)

    for point in dataset:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return zip(assignments, dataset)

class node_vec():
    def __init__(self, vec=None):
        self.vec = vec
        if vec is None:
            self.size = 0
        else:
            self.size = 1
    def verb_num(self):
        return len(self.vec)
    def insert(self,point):
        if self.vec is None:
            self.vec = copy.deepcopy(point.vec)
        else:
            self.vec += point.vec
        self.size += 1

    def mean(self):
        return self.vec / self.size

    def __str__(self):
        return str(self.mean())


class node():
    def __init__(self, verbs):
        self.verbs = {}
        for verb in verbs:
            if verb not in self.verbs:
                self.verbs[verb] = 0
            self.verbs[verb] += 1
        self.num_verbs = len(self.verbs)

    def distance_euclid(self, target):
        ret = 0.0
        norm1 = self.norm()
        norm2 = target.norm()
        if norm1 == 0 or norm2 == 0:
            return 10000
        for verb in self.verbs:
            x1 = self.verbs[verb] / norm1
            if verb in target.verbs:
                x2 = target.verbs[verb] / norm2
            else:
                x2 = 0
            ret += (x1 - x2) ** 2
        for verb in target.verbs:
            if verb not in self.verbs:
                ret += (target.verbs[verb] / norm2) ** 2
        return ret/2

    def distance(self, target):
        return self.distance_euclid(target)
        fz = 0.0
        fm = self.norm() * target.norm()
        for verb in self.verbs:
            if verb in target.verbs:
                fz += self.verbs[verb] * target.verbs[verb]
        if fm == 0.0:
            return 1.0
        return 1.0 - fz / fm

    def distance_distinct(self, point):
        ret = 0.0
        for verb in point.verbs:
            if verb not in self.verbs:
                ret += 1.0
        for verb in self.verbs:
            if verb not in point.verbs:
                ret += 1.0
        return ret

    def distance_new_distinct(self, point):
        #if len(point.verbs)==0:
        #    return 10000
        ret = 0.0
        for verb in point.verbs:
            if verb not in self.verbs:
                ret += 1.0
        return ret

    def remove(self, ex_vi):
        for vi in ex_vi.verbs:
            self.verbs[vi] -= 1
            if self.verbs[vi] == 0:
                self.verbs.pop(vi)

    def insert(self, ex_vi):
        for vi in ex_vi.verbs:
            if vi not in self.verbs:
                self.verbs[vi] = 0
            self.verbs[vi] += 1

    def norm(self):
        ret = 0.0
        for verb in self.verbs:
            ret += self.verbs[verb] * self.verbs[verb]
        return sqrt(ret)

    def overlap_ratio(self, point):
        if len(self.verbs) == 0:
            return 0.0
        overlap = 0.0
        for verb in point.verbs:
            if verb in self.verbs:
                overlap += 1.0
        return overlap / len(self.verbs)

    def __str__(self):
        return 'Num:{} {}'.format(len(self.verbs),str(self.verbs))


def update_centers_balanced(dataset, assignments, is_node = True):
    centers = []
    for i in range(max(assignments)+1):
        if is_node:
            centers.append(node([]))
        else:
            centers.append(node_vec())

    for assignment, point in zip(assignments, dataset):
        if assignment != -1:
            centers[assignment].insert(point)

    return centers

def convert_vec_list_to_tensor(a):
    ret = []
    for t in a:
        ret.append(t.vec)
    return torch.from_numpy(np.array(ret))

def qsort(ls):
    if len(ls) > 1:
        pivot = ls[0][0]
        left = [e for e in ls if e[0] < pivot]
        equal = [e for e in ls if e[0] == pivot]
        right = [e for e in ls if e[0] > pivot]
        return qsort(left) + equal + qsort(right)
    else:
        return ls

def assign_points_balanced(epoch, dataset, centers, batch_size, old_assignments, is_node):
    assignments = [-1] * len(dataset)
    dis = []
    shuffled_list = list(range(len(dataset)))
    random.shuffle(shuffled_list)

    num_point = len(dataset)
    num_center = len(centers)
    if is_node == False:
        pair_dis = torch.cdist(convert_vec_list_to_tensor(dataset).unsqueeze(0), convert_vec_list_to_tensor(centers).unsqueeze(0))
        print('computed pair dis')
        pair_dis = pair_dis[0].reshape(-1,1)
        index_points = torch.arange(num_point).unsqueeze(-1).expand(-1,num_center).reshape(-1,1)
        index_centers = torch.arange(num_center).unsqueeze(0).expand(num_point,-1).reshape(-1,1)
        dis = torch.cat([pair_dis,index_points,index_centers],dim=-1)
        print('prepared pair dis')

        '''for i in tqdm(range(len(dataset))):
            for j in range(len(centers)):
                dis.append([pair_dis[0][i][j],i,j])'''
    else:
        for i, point in tqdm(enumerate(dataset)):
            for j, center in enumerate(centers):
                dis.append([distance(center,point),i,j])
                #dis.append([center.distance_euclid(point), i, j])
                #dis.append([- center.overlap_ratio(point), i, j])
                #dis.append([center.distance_new_distinct(point), i, j])
                #dis.append([- center.overlap_ratio(point) * len(point.verbs), i, j])
                #dis.append([(1.0 - center.overlap_ratio(point)) * len(point.verbs), i, j])
                #dis.append([center.distance_new_distinct(point)-center.overlap_ratio(point),i,j])
                #dis.append([- center.overlap_ratio(point) * len(center.verbs) * len(point.verbs), i, j])
                #dis.append([- center.overlap_ratio(point) * len(center.verbs), i, j])
                #dis.append([center.distance_new_distinct(point) - center.overlap_ratio(point) * len(center.verbs), i,j])
                #dis.append([center.distance_new_distinct(point) - center.overlap_ratio(point) * len(point.verbs), i, j])
                #dis.append([center.distance_new_distinct(point) + center.distance_euclid(point), i, j])
                #dis.append([center.distance_euclid(point) / math.pow(len(center.verbs),0.1), i, j])
                #dis.append([center.distance_distinct(point), i, j])
    #dis = qsort(dis)
    #dis, _ = torch.sort(dis,dim=-1)
    #dis.sort(dim=-1)
    #dis.sort(key=lambda x: x[0])
    start_time = time.time()
    dis = dis.cpu()
    dis = dis[dis[:, 0].argsort()]
    dis = dis.cpu().detach().numpy()
    print("---sorted in %.4f seconds ---" % (time.time() - start_time))
    cluster_size = [0] * len(centers)
    cluster_indexes = []
    for i in range(len(centers)):
        cluster_indexes.append([])

    assigned_num = 0
    non_empty_cluster_num = 0
    i_list = dis[:,1].astype(int)
    j_list = dis[:,2].astype(int)
    d_list = dis[:,0]

    #for d, i, j in tqdm(dis):
    # print(len(i_list), len(j_list), len(d_list))
    for index in tqdm(range(len(i_list))):
        if assignments[i_list[index]] == -1 and cluster_size[j_list[index]] < batch_size:

            assigned_num += 1
            assignments[i_list[index]] = j_list[index]
            if cluster_size[j_list[index]] == 0:
                non_empty_cluster_num += 1
            cluster_size[j_list[index]] += 1
            cluster_indexes[j_list[index]].append(i)
            if assigned_num == num_point:
                break
    return assignments, cluster_indexes

def distance(center,point):
    if isinstance(center,node_vec):
        return np.linalg.norm(center.mean()-point.vec)
    #elif len(point.verbs) == 0:
    #    return 10000
    else:
        #return center.distance_new_distinct(point)/len(point.verbs)
        return - center.overlap_ratio(point)
        #return center.distance_euclid(point)

def assign_points_balanced_heap(epoch, ex_vi, centers, batch_size, old_assignments):
    assignments = [-1] * len(ex_vi)
    heap = MinHeap()
    first_zero = True
    cluster_num = len(centers)

    for i, point in enumerate(ex_vi):
        for j, center in enumerate(centers):
            ij = '{}_{}'.format(i,j)

            d = distance(center,point)

            heap.insert(d, ij)
    assigned_num = 0
    cluster_size = [0] * len(centers)
    non_empty_cluster_num = 0
    cluster_indexes = []
    for i in range(len(centers)):
        cluster_indexes.append([])
    assigned_dis = [0] * len(assignments)
    while assigned_num < len(centers) * batch_size:
        d, ij = heap.delete_min()
        i, j = [int(t) for t in ij.split('_')]
        if assignments[i] == -1 and cluster_size[j]<batch_size:
            assigned_num += 1
            assignments[i] = j
            assigned_dis[i] = d
            if cluster_size[j] == 0:
                non_empty_cluster_num += 1
            cluster_size[j] += 1
            cluster_indexes[j].append(i)
            centers[j].insert(ex_vi[i])

            if d>=1-1e-8 and first_zero and epoch>=1:
                print('assigned num: {}/{}, non_empty/total cluster: {}/{}'.format(assigned_num,len(ex_vi),non_empty_cluster_num,len(centers)))
                first_zero = False

            for i1 in range(len(assignments)):
                if assignments[i1] == -1 and centers[j].overlap_ratio(ex_vi[i1])>0:
                    d = distance(centers[j], ex_vi[i1])
                    i1j = '{}_{}'.format(i1, j)
                    heap_index = heap.ij_index[i1j]
                    heap.remove(heap_index)
                    heap.insert(d,i1j)

    return assignments, cluster_indexes


def generate_k_balanced(dataset, k, is_node):
    centers = []
    for i in range(k):
        index = random.randint(0, len(dataset) - 1)
        #while (len(dataset[index].verbs) == 0):
        #    index = random.randint(0, len(dataset) - 1)
        if is_node:
            centers.append(node(dataset[index].verbs))
        else:
            centers.append(node_vec(dataset[index].vec))

    return centers

def random_assign(ex_vi, k, batch_size):
    num = len(ex_vi)
    indexes = list(range(num))
    random.shuffle(indexes)
    cluster_index_now = 0
    cluster_size = [0]*k
    assignments = [-1] * num
    for i in indexes:
        if cluster_size[cluster_index_now]==batch_size:
            cluster_index_now += 1
        if cluster_index_now>=k:
            break
        cluster_size[cluster_index_now]+=1
        assignments[i]=cluster_index_now
    return assignments



def generate_k_balanced_opt(dataset, k, is_node):
    centers = []
    used = {}
    for i in range(k):
        p = []
        for j in range(len(dataset)):
            if j in used or (is_node and len(dataset[j].verbs)) == 0:
                p.append(0)
                continue
            min_dis = 1e3
            for center in centers:
                t_dis = distance(center,dataset[j]) #center.distance_euclid(dataset[j])
                if t_dis < min_dis:
                    min_dis = t_dis
            p.append(min_dis ** 2)
        index = random.choices(range(len(dataset)), p)[0]
        if is_node:
            centers.append(node(dataset[index].verbs))
        else:
            centers.append(node_vec(dataset[index].vec))

    return centers


def count_distinct(ex_vi, assignments):
    cluster_verb = {}
    for i, assignment in enumerate(assignments):
        if assignment not in cluster_verb:
            cluster_verb[assignment] = {}
        for verb in ex_vi[i]:
            cluster_verb[assignment][verb] = 1

    total_distinct = []
    non_empty_cluster_num = 0
    for cluster in cluster_verb:
        verbs = {}
        for verb in cluster_verb[cluster]:
            verbs[verb] = 1
        total_distinct.append(len(verbs))
        if len(verbs)>0:
            non_empty_cluster_num +=1

    return np.mean(total_distinct), np.std(total_distinct)


def swap_all_best_only(ex_vi, assignments):
    clusters = []
    for i in range(max(assignments) + 1):
        clusters.append(node([]))
    for i, ci in enumerate(assignments):
        clusters[ci].insert(ex_vi[i])

    for i in range(len(assignments)):
        max_reduce = 0
        best_j = None
        for j in range(i + 1, len(assignments)):
            if assignments[i] != assignments[j]:
                ci = assignments[i]
                cj = assignments[j]
                t1 = len(clusters[ci].verbs) + len(clusters[cj].verbs)
                clusters[ci].remove(ex_vi[i])
                clusters[cj].remove(ex_vi[j])

                clusters[ci].insert(ex_vi[j])
                clusters[cj].insert(ex_vi[i])
                t2 = len(clusters[ci].verbs) + len(clusters[cj].verbs)
                if t1 - t2 > max_reduce:
                    max_reduce = t1 - t2
                    best_j = j
                clusters[ci].remove(ex_vi[j])
                clusters[cj].remove(ex_vi[i])

                clusters[ci].insert(ex_vi[i])
                clusters[cj].insert(ex_vi[j])

        if max_reduce > 0:
            cj = assignments[best_j]
            clusters[ci].remove(ex_vi[i])
            clusters[cj].remove(ex_vi[best_j])

            clusters[ci].insert(ex_vi[best_j])
            clusters[cj].insert(ex_vi[i])
            temp = assignments[i]
            assignments[i] = assignments[best_j]
            assignments[best_j] = temp

    return assignments


def swap_all(ex_vi, assignments):
    clusters = []
    for i in range(max(assignments) + 1):
        clusters.append(node([]))
    for i, ci in enumerate(assignments):
        clusters[ci].insert(ex_vi[i])

    for i in range(len(assignments)):
        for j in range(i+1, len(assignments)):
            if assignments[i] != assignments[j]:
                ci = assignments[i]
                cj = assignments[j]
                t1 = len(clusters[ci].verbs) + len(clusters[cj].verbs)
                clusters[ci].remove(ex_vi[i])
                clusters[cj].remove(ex_vi[j])

                clusters[ci].insert(ex_vi[j])
                clusters[cj].insert(ex_vi[i])
                t2 = len(clusters[ci].verbs) + len(clusters[cj].verbs)
                if t1 > t2:
                    temp = assignments[i]
                    assignments[i] = assignments[j]
                    assignments[j] = temp
                else:
                    clusters[ci].remove(ex_vi[j])
                    clusters[cj].remove(ex_vi[i])

                    clusters[ci].insert(ex_vi[i])
                    clusters[cj].insert(ex_vi[j])

    return assignments

def swap_all_cluster_wise(ex_vi, assignments):
    clusters = []
    cluster_indexes = []
    cluster_num = max(assignments) + 1
    for i in range(cluster_num):
        clusters.append(node([]))
        cluster_indexes.append([])
    for i, ci in enumerate(assignments):
        clusters[ci].insert(ex_vi[i])
        cluster_indexes[ci].append(i)

    for i in range(len(assignments)):
        ci = assignments[i]
        for cj1 in range(cluster_num):
            max_reduce = 0
            best_j = None
            for j in cluster_indexes[cj1]:
                cj = assignments[j]
                if assignments[i] != assignments[j]:
                    t1 = len(clusters[ci].verbs) + len(clusters[cj].verbs)
                    clusters[ci].remove(ex_vi[i])
                    clusters[cj].remove(ex_vi[j])

                    clusters[ci].insert(ex_vi[j])
                    clusters[cj].insert(ex_vi[i])
                    t2 = len(clusters[ci].verbs) + len(clusters[cj].verbs)
                    if t1 - t2 > max_reduce:
                        max_reduce = t1 - t2
                        best_j = j
                    clusters[ci].remove(ex_vi[j])
                    clusters[cj].remove(ex_vi[i])

                    clusters[ci].insert(ex_vi[i])
                    clusters[cj].insert(ex_vi[j])

            if max_reduce > 0:
                cj = assignments[best_j]
                clusters[ci].remove(ex_vi[i])
                clusters[cj].remove(ex_vi[best_j])
                clusters[ci].insert(ex_vi[best_j])
                clusters[cj].insert(ex_vi[i])
                temp = assignments[i]
                assignments[i] = assignments[best_j]
                assignments[best_j] = temp

    return assignments

def count_non_empty_point(ex_vi):
    num = 0
    verbs = {}
    for point in ex_vi:
        if len(point.verbs)>0:
            num += 1
        for verb in point.verbs:
            verbs[verb] = 1
    print('non empty points: {}'.format(num))
    return num, len(verbs)

def count_join(ex_vi, cluster_indexes):
    avg1 = 0
    avg2 = 0
    avg3 = 0
    avg4 = 0
    for indexes in cluster_indexes:
        sum2 = 0
        sum3 = 0
        sum1 = 0
        n = len(indexes)
        for i in range(n):
            sum1 += len(ex_vi[indexes[i]])
            for j in range(i+1,n):
                for vi in ex_vi[indexes[i]]:
                    if vi in ex_vi[indexes[j]]:
                        sum2 +=1
                for k in range(j+1,n):
                    for vi in ex_vi[indexes[i]]:
                        if vi in ex_vi[indexes[j]] and vi in ex_vi[indexes[k]]:
                            sum3 +=1
                    for l in range(k+1,n):
                        for vi in ex_vi[indexes[i]]:
                            if vi in ex_vi[indexes[j]] and vi in ex_vi[indexes[k]] and vi in ex_vi[indexes[l]]:
                                avg4 +=1
        avg1 += sum1
        avg2 += sum2
        avg3 += sum3
    print('avg1:{:.5f}   avg2:{:.5f}   avg3:{:.5f}   avg4:{:.5f}'.format(avg1 / len(cluster_indexes),
                                                                         avg2 / len(cluster_indexes),
                                                                         avg3 / len(cluster_indexes),
                                                                         avg4 / len(cluster_indexes)))
    return avg2/len(cluster_indexes), avg3/len(cluster_indexes)

def compute_avg_dis(centers, dataset, assignments):
    dis = []
    for i in range(len(assignments)):
        if assignments[i]!=-1:
            c = assignments[i]
            dis.append(distance(centers[c],dataset[i]))
    return np.sum(dis) / len(dis), np.std(dis)


def balanced_k_means(ex_vi, batch_size, is_node = True, ex_vi_ori = None, max_epoch = None, logger=None):
    num = len(ex_vi)
    k = floor(num / batch_size)
    dataset = []

    avg_distance = []
    sd_distance = []
    avg_distinct = []
    sd_distinct = []
    for i, vi in enumerate(ex_vi):
        if is_node:
            dataset.append(node(vi))
        else:
            dataset.append(node_vec(copy.deepcopy(vi)))

    if is_node:
        non_empty_num, distinct_verb_num = count_non_empty_point(dataset)

        logger.info('lower bound: {}'.format(distinct_verb_num / (len(ex_vi) / batch_size)))
        random_assignments = random_assign(ex_vi,k,batch_size)
        logger.info('random assigned distinct {}'.format(count_distinct(ex_vi, random_assignments)[0]))
    else:
        random_assignments = random_assign(ex_vi_ori, k, batch_size)
        new_centers = update_centers_balanced(dataset, random_assignments, is_node)
        a, b = compute_avg_dis(new_centers, dataset, random_assignments)
        c, d = count_distinct(ex_vi_ori, random_assignments)

        logger.info('random assigned distinct {}'.format(a))
        logger.info('random avg dis to center: {}'.format(c))
        avg_distance.append(a)
        sd_distance.append(b)
        avg_distinct.append(c)
        sd_distinct.append(d)
    # init_centers = generate_k_balanced(dataset, k, is_node)
    #init_centers = generate_k_balanced_opt(dataset,k,is_node)
    assignments, cluster_indexes = assign_points_balanced(-1,dataset, new_centers, batch_size, old_assignments = None, is_node = is_node)
    new_centers = update_centers_balanced(dataset, assignments, is_node)
    
    epoch = 0
    # if is_node:
    #     logger.info('epoch: {} avg distinct: {}'.format(epoch, count_distinct(ex_vi, assignments)))
    # else:
    #     logger.info('epoch: {} avg distinct: {}'.format(epoch, count_distinct(ex_vi_ori, assignments)))
    #     logger.info('avg dis to center: {}'.format(compute_avg_dis(new_centers, dataset, assignments)))

    old_assignments = None
    while assignments != old_assignments and (max_epoch is None or epoch < max_epoch):
        old_assignments = copy.deepcopy(assignments)
        assignments, cluster_indexes = assign_points_balanced(epoch, dataset, new_centers, batch_size,
                                                              old_assignments=old_assignments, is_node = is_node)
        new_centers = update_centers_balanced(dataset, assignments, is_node)
        epoch += 1
        if is_node:
            logger.info('epoch: {} avg distinct: {}'.format(epoch, count_distinct(ex_vi, assignments)))
        else:
            a, b = compute_avg_dis(new_centers, dataset, assignments)
            c, d = count_distinct(ex_vi_ori, assignments)
            # logger.info('epoch: {} avg distinct: {}'.format(epoch, count_distinct(ex_vi_ori, assignments)))
            # logger.info('avg dis to center: {}'.format(compute_avg_dis(new_centers, dataset, assignments)))
            # print('epoch: {} avg distinct: {}'.format(epoch, count_distinct(ex_vi_ori, assignments)))
            # print('avg dis to center: {}'.format(compute_avg_dis(new_centers, dataset, assignments)))
            avg_distance.append(a)
            sd_distance.append(b)
            avg_distinct.append(c)
            sd_distinct.append(d)
        #assignments = swap_all(dataset,assignments)
        #print('after swap: {} avg distinct: {}'.format(epoch, count_distinct(ex_vi, assignments)))
        #count_distinct(ex_vi, assignments)
    if num % batch_size != 0:
        k += 1
        for i in range(len(assignments)):
            if assignments[i] == -1:
                assignments[i] = k - 1

    return assignments, k, zip(avg_distance, sd_distance, avg_distinct, sd_distinct)
    # return assignments, k

def compute_affinity_graph(ex_vi):
    graph_size = len(ex_vi)
    ret = np.zeros((graph_size,graph_size))
    for i,verbs1 in enumerate(ex_vi):
        for j, verbs2 in enumerate(ex_vi):
            if i!=j:
                for vi in ex_vi[i]:
                    if vi in ex_vi[j]:
                        ret[i][j]+=1
    return ret

def compute_affinity_graph_sparse(ex_vi):
    def insert_edge(graph, a, b, w):
        if a not in graph:
            graph[a] = {}
        if b not in graph[a]:
            graph[a][b] = 0
        graph[a][b]+=w
    graph_size = len(ex_vi)
    graph = {}
    vi_i = {}
    # print(ex_vi)
    for i,verbs in enumerate(ex_vi):
        for vi in verbs:
            if vi not in vi_i:
                vi_i[vi] = []
            vi_i[vi].append(i)

    with tqdm(total=len(ex_vi)) as pbar:
        for i, verbs in enumerate(ex_vi):
            for vi in verbs:
                for j in vi_i[vi]:
                    if i!=j:
                        insert_edge(graph, i, j, 1)
                        insert_edge(graph, j, i, 1)
            pbar.update(1)

    x_list = []
    y_list = []
    w_list = []
    for a in graph:
        for b,w in graph[a].items():
            x_list.append(a)
            y_list.append(b)
            w_list.append(w)

    return csr_matrix((w_list, (x_list, y_list)), shape=(graph_size, graph_size))

from sklearn.manifold import spectral_embedding
from scipy.linalg import eigh
def compute_spectral_embedding(affinity, n_components, eigen_solver ='amg', method = 'ng'):
    print('compute spectral embedding for {} nodes'.format(affinity.shape[0]))
    if method == 'ng':
        ret = spectral_embedding(affinity, n_components=n_components, eigen_solver=eigen_solver, drop_first=False,
                                  norm_laplacian=True)
        ret = scikit.preprocessing.normalize(ret,axis =1)
    elif method == 'shi':
        D = np.zeros(affinity.shape)
        sum_raw = np.sum(affinity,axis = 1)
        for i in range(affinity.shape[0]):
            D[i][i] = sum_raw[i]
        eigvals, eigvecs = eigh(affinity, D, eigvals_only=False)
        eigvecs = eigvecs.real[np.argsort(eigvals)]
        eigvecs = eigvecs[:,:n_components]
        return eigvecs


    return ret


fu = []
def get_fu(i):
    if fu[i] == i:
        return
    get_fu(fu[i])
    fu[i]=fu[fu[i]]

def add_connectivity(affinity):
    n = affinity.shape[0]
    for i in range(n):
        fu.append(i)
    for i in range(n):
        for j in range(n):
            if affinity[i][j]>0:
                get_fu(i)
                get_fu(j)
                fu[fu[i]] = fu[j]
    for i in range(n):
        get_fu(i)
    for i in range(n-1):
        if fu[i]!=fu[i+1]:
            affinity[i+1][i] = 1
            affinity[i][i +1] = 1
    return affinity


def balanced_spectrum(ex_vi, batch_size, logger):
    num = len(ex_vi)
    k = floor(num / batch_size)
    #affinity = compute_affinity_graph(ex_vi)
    affinity = compute_affinity_graph_sparse(ex_vi)
    #affinity = add_connectivity(affinity)
    '''affinity =np.array([[0., 1., 1., 0., 0., 1., 0., 0., 1., 1.],
                [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 1., 1., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
    D = np.diag(affinity.sum(axis=1))
    L = D - affinity
    vals, vecs = np.linalg.eig(L)
    vecs = vecs[:, np.argsort(vals)]
    vals = vals[np.argsort(vals)]'''

    start_time = time.time()
    maps = compute_spectral_embedding(affinity,n_components = 8)
    logger.info("---compute spectral embedding %.4f seconds ---" % (time.time() - start_time))
    return balanced_k_means(maps, batch_size, is_node = False, ex_vi_ori=ex_vi,max_epoch=4, logger=logger)
    '''assignments, _ = kmeans(torch.from_numpy(maps), k+1, batch_size, max_iter=10)
    print('avg distinct: {}'.format(count_distinct(ex_vi, assignments.int().numpy())))
    return assignments'''

def paritition(ex_vi, batch_size):
    return balanced_spectrum(ex_vi, batch_size)
    #return balanced_k_means(ex_vi, batch_size, max_epoch=3)
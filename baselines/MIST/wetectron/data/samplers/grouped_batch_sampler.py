import itertools
import random
import torch
import numpy as np

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from wetectron.data.datasets import PascalVOCDataset, COCODataset

from collections import Counter


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``
    """

    def __init__(self, sampler, group_ids, batch_size, b_size, dataset, class_batch, data_args = None, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

        if class_batch and len(data_args[0]) == 2:
            self.voc_train = PascalVOCDataset(**data_args[0][0])
            self.voc_val = PascalVOCDataset(**data_args[0][1])
            self.dataset_type = 'voc'
        elif class_batch and len(data_args[0]) == 1:
            self.coco_train = COCODataset(**data_args[0][0])
            self.dataset_type = 'coco'
        self.dataset = dataset

        self.class_batch = class_batch
        self.b_size = b_size

    def get_img_labels(self, index):
        if self.dataset.get_idxs(index)[0] == 0:
            img_labels = self.voc_train.get_groundtruth(
                self.dataset.get_idxs(index)[1]).get_field('labels')
        else:
            img_labels = self.voc_val.get_groundtruth(
                self.dataset.get_idxs(index)[1]).get_field('labels')
        return img_labels

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutatin between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept

        if self.class_batch:

            class_labels = []
            if self.dataset_type == 'voc':
                for d in range(len(self.dataset)):
                    class_labels.append(list(set(self.get_img_labels(d).tolist())))
            elif self.dataset_type == 'coco':
                for d in range(len(self.dataset)):
                    class_labels.append(list(set(self.dataset.get_groundtruth(d).tolist())))

            #if self.dataset_type == 'voc':
            #    for d in range(len(self.dataset)):
            #        class_labels.append(list(set(self.get_img_labels(d).tolist())))
            #elif self.dataset_type == 'coco':
            #    for d in range(len(self.dataset)):
            #        class_labels.append(list(set(self.dataset.get_groundtruth(d).tolist())))


            ### co_occurrence ###
            count = [0]*20
            for cls in class_labels:
                for c in cls:
                    count[c-1] += 1
            ### co_occurrence count objects by class ###

            share_labels = [[] for x in range(20)]
            share_with_neg = [[] for x in range(20)]
            share_single = [[] for x in range(20)]
            for i, cls in enumerate(class_labels):
                for c in cls:
                    for j in range(20):
                        if c-1 == j:
                            share_labels[j].append(i)
                            if len(cls) == 1:
                                share_single[j].append(i)
                            elif len(cls) >= 2:
                                share_with_neg[j].append(i) ### reference image only contain neg

            sampled_id = sampled_ids.tolist().copy()
            inds = sampled_ids.tolist().copy()
            sample_id = sampled_ids.tolist().copy()
            batch = []
            share_c = [0]*20

            ###
            '''
            np_class_labels = np.array(class_labels)
            while np_class_labels[sample_id].shape != np.unique(np_class_labels[sample_id]).shape:
                not_selected = True
                while not_selected:
                    rand_ind1 = random.choice(sample_id)
                    rand_class1 = random.choice(class_labels[rand_ind1])
                    rand_ind2 = random.choice(sample_id)
                    rand_class2 = random.choice(class_labels[rand_ind2])
                    if rand_class1 in class_labels[rand_ind2] and rand_ind1 != rand_ind2:
                        if class_labels[rand_ind1] == class_labels[rand_ind2] and len(class_labels[rand_ind1]) > 1 and len(class_labels[rand_ind2]) > 1:
                            not_selected = True
                        else:
                            batch.append([rand_ind1, rand_ind2])
                            not_selected = False
                            sample_id.remove(rand_ind1)
                            sample_id.remove(rand_ind2)
                            #sample_id = np.delete(sample_id, np.where(sample_id == rand_ind1))
                            #sample_id = np.delete(sample_id, np.where(sample_id == rand_ind2))
            return batch
            '''

            for i, ind1 in enumerate(inds):
                rand_class1 = np.random.choice(class_labels[ind1])
                for j, ind2 in enumerate(inds[i+1:]):
                    if rand_class1 in class_labels[ind2] and class_labels[ind1] != class_labels[ind2] \
                        or rand_class1 in class_labels[ind2] and class_labels[ind1] == class_labels[ind2] and len(class_labels[ind1])==1:
                        #or rand_class1 in class_labels[ind2] and class_labels[ind1] == class_labels[ind2] and len(class_labels[ind1])==1:
                        batch.append([ind1, ind2])
                        inds.remove(ind2)
                        break
            return batch

            '''
            ### step1 ###
            for i, s_i in enumerate(sampled_id):
                rand_c = random.choice(class_labels[s_i])
                for j, r in enumerate(sampled_id[i+1:]):
                    if rand_c in class_labels[r] and self.group_ids[s_i] == self.group_ids[r]:
                        batch.append([s_i, r])
                        sampled_id.remove(r)
                        break
                share_c[rand_c-1] += 1
            ### step1 ###
            ### step2 ###
            cross_batch = [y for x in batch for y in x]
            original_batch = [y for x in batches for y in x]
            diff = list((Counter(original_batch) - Counter(cross_batch)).elements())
            sampled_id2 = sampled_ids.tolist().copy()
            for i, d in enumerate(diff):
                rand_c = random.choice(class_labels[d])
                for j, r in enumerate(sampled_id2[i+1:]):
                    if rand_c in class_labels[r] and self.group_ids[d] == self.group_ids[r]:
                        batch.append([d, r])
                        sampled_id2.remove(r)
                        break
            ### step2 ###
            ### step3 ###
            batch = batch[:len(batches)]
            ### step3 ###
            '''
            '''
            share_count = [0]*20
            for i,b in enumerate(batches):
                rand = random.choice(class_labels[b[0]])
                #rand_choice = random.choice(share_labels[rand-1])       ### totally random
                rand_choice = random.choice(share_with_neg[rand-1])    ### with neg class
                share_count[rand-1] += 1                                ### count shared class
                #if len(class_labels[b[0]]) == 1:
                #    rand_choice = random.choice(share_with_neg[rand-1])
                #else :
                #    rand_choice = random.choice(share_single[rand-1])
                #batches[i].append(rand_choice)

                if b[0] != rand_choice:
                    batches[i].append(rand_choice)
                elif b[0] == rand_choice:
                #    batches[i].append(random.choice(share_labels[rand-1]))
                    batches[i].append(random.choice(share_with_neg[rand-1]))
            return batches
            '''
            return batch
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
